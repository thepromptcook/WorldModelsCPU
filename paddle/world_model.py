"""A Genie-style world model of the two-player paddle game (32x32), trained on a CPU.

Input: the last 2 frames and BOTH players' key presses. Output: the next frame, generated
MaskGIT-style (see README: why generate cell by cell) so random serves are sampled, not averaged away.
Training uses the improved recipe: varied players, 4-way symmetry augmentation (mirroring
swaps the players), prioritized replay, weight averaging (EMA), bfloat16, time budget.

    THREADS=4 python paddle_world_model.py train --minutes 50 --episodes 4000
    python paddle_world_model.py eval      # rule-check dreams + paddle_dream.gif
    python paddle_world_model.py test      # check the symmetry augmentation is exact
"""
import copy
import json
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from game import (W, H, P, LEFT, RIGHT, BALL, BALL_VEL, N_CLASSES, N_ACTIONS, PaddleGame,
                         bot, render, font, is_ball)

K = 2
MASK = N_CLASSES
torch.set_num_threads(int(os.environ.get("THREADS", 4)))


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.norm = nn.GroupNorm(8, w)
        self.c1 = nn.Conv2d(w, w, 3, padding=1)
        self.glob = nn.Sequential(nn.Linear(2 * w, w), nn.GELU(), nn.Linear(w, w))
        self.c2 = nn.Conv2d(w, w, 3, padding=1)

    def forward(self, x):
        h = self.c1(F.gelu(self.norm(x)))
        h = h + self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))[..., None, None]
        return x + self.c2(F.gelu(h))


class PaddleWorldModel(nn.Module):
    def __init__(self, w=64, depth=6):
        super().__init__()
        self.inp = nn.Conv2d(K * N_CLASSES + N_CLASSES + 1 + 2 * N_ACTIONS, w, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(w) for _ in range(depth)])
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.GELU(), nn.Conv2d(w, N_CLASSES, 1))

    def forward(self, hist, actions, partial):
        """hist (B,K,H,W), actions (B,2) = (left, right), partial (B,H,W) with MASK."""
        B = hist.shape[0]
        h = F.one_hot(hist, N_CLASSES).permute(0, 1, 4, 2, 3).reshape(B, K * N_CLASSES, H, W)
        p = F.one_hot(partial, N_CLASSES + 1).permute(0, 3, 1, 2)
        a = F.one_hot(actions, N_ACTIONS).reshape(B, 2 * N_ACTIONS)[..., None, None].expand(B, 2 * N_ACTIONS, H, W)
        return self.out(self.blocks(self.inp(torch.cat([h, p, a], 1).float())))

    @torch.no_grad()
    def generate(self, hist, actions, gen=None, top_p=0.95, commit_above=0.999):
        """One frame, MaskGIT-style. Returns (frame, passes)."""
        frame = torch.full((H, W), MASK, dtype=torch.long)
        flat, masked = frame.view(-1), torch.ones(H * W, dtype=torch.bool)
        passes = 0
        while masked.any():
            passes += 1
            logits = self(hist[None], torch.tensor([actions]), frame[None])[0]
            probs = logits.permute(1, 2, 0).reshape(-1, N_CLASSES).softmax(-1)
            sp, si = probs.sort(-1, descending=True)
            filtered = torch.zeros_like(probs).scatter(-1, si, sp * ((sp.cumsum(-1) - sp) < top_p))
            sample = torch.multinomial(filtered, 1, generator=gen)[:, 0]
            conf = probs.gather(-1, sample[:, None])[:, 0]
            conf[~masked] = -1
            m = int(masked.sum())
            keep = conf.topk(max(int((conf > commit_above).sum()), (m + 1) // 2 if m > 8 else 1)).indices
            flat[keep], masked[keep] = sample[keep], False
        return frame, passes


# ---------- data ----------

def make_data(n_episodes, seed, length=300):
    """Two bots play; each episode picks a style and carelessness for each side."""
    rng = np.random.default_rng(seed)
    chunks, hidx, nidx, acts, rare, total = [], [], [], [], [], 0
    offsets = np.arange(K - 1, -1, -1)
    for _ in range(n_episodes):
        game = PaddleGame(rng)
        styles = rng.choice(4, size=2, p=[0.4, 0.3, 0.15, 0.15])
        eps = rng.uniform(0, 0.3, size=2)
        ep, ep_acts, events = [game.frame()], [], []
        for _ in range(length):
            a = (bot(ep[-1], 0, rng, styles[0], eps[0]), bot(ep[-1], 1, rng, styles[1], eps[1]))
            (r, c), vel = game.ball, game.vel
            ep.append(game.step(*a))
            ep_acts.append(a)
            events.append(game.vel[1] != vel[1] or c in (0, W - 1) or game.ball[1] in (0, W - 1))   # hit, miss, serve
        t = np.arange(len(ep_acts))
        chunks.append(np.array(ep, np.uint8))
        hidx.append(total + np.maximum(t[:, None] - offsets[None, :], 0))
        nidx.append(total + t + 1)
        acts.append(np.array(ep_acts))
        rare.append(np.array(events))
        total += len(ep)
    cat = lambda xs: torch.from_numpy(np.concatenate(xs))
    return cat(chunks), cat(hidx), cat(nidx), cat(acts).long(), cat(rare)


# ---------- symmetry augmentation: vertical mirror, horizontal mirror (swaps players), both ----------

def symmetries():
    out = []
    for vflip in (False, True):
        for hflip in (False, True):
            cls = torch.arange(N_CLASSES)
            for (dr, dc), k in BALL.items():
                cls[k] = BALL[(-dr if vflip else dr, -dc if hflip else dc)]
            if hflip:
                cls[LEFT], cls[RIGHT] = RIGHT, LEFT
            act = torch.tensor([0, 2, 1]) if vflip else torch.arange(N_ACTIONS)   # up <-> down
            out.append((vflip, hflip, cls, act))
    return out


def transform(x, sym):
    vflip, hflip, cls, _ = sym
    if vflip:
        x = x.flip(-2)
    if hflip:
        x = x.flip(-1)
    return cls[x]


def transform_actions(a, sym):
    vflip, hflip, _, act = sym
    a = act[a]
    return a.flip(-1) if hflip else a                 # mirrored left/right: the players swap


def test_symmetries(trials=200):
    rng = np.random.default_rng(0)
    for trial, sym in zip(range(trials), symmetries() * trials):
        vflip, hflip = sym[0], sym[1]
        g1 = PaddleGame(rng)
        g2 = copy.copy(g1)

        def mirror(g_src, g_dst):
            (r, c), (dr, dc) = g_src.ball, g_src.vel
            g_dst.ball = (H - 1 - r if vflip else r, W - 1 - c if hflip else c)
            g_dst.vel = (-dr if vflip else dr, -dc if hflip else dc)
            lt, rt = g_src.left, g_src.right
            if vflip:
                lt, rt = H - P - lt, H - P - rt
            g_dst.left, g_dst.right = (rt, lt) if hflip else (lt, rt)

        mirror(g1, g2)
        for _ in range(200):
            f1 = torch.from_numpy(g1.frame())
            assert torch.equal(transform(f1, sym), torch.from_numpy(g2.frame())), "frame mismatch"
            a = torch.tensor([int(rng.integers(0, 3)), int(rng.integers(0, 3))])
            serving = g1.ball[1] in (0, W - 1)
            g1.step(*a.tolist())
            to = None
            if serving:
                (r, c), (dr, dc) = g1.ball, g1.vel
                to = (H - 1 - r if vflip else r, W - 1 - c if hflip else c, -dr if vflip else dr, -dc if hflip else dc)
            g2.step(*transform_actions(a, sym).tolist(), serve_to=to)
        assert torch.equal(transform(torch.from_numpy(g1.frame()), sym), torch.from_numpy(g2.frame()))
    print(f"symmetry augmentation exact on {trials} games x 200 steps, all 4 symmetries")


# ---------- training ----------

def one_cycle(progress, lr_max=2e-3, pct=0.3, div=25, final_div=1e4):
    anneal = lambda start, end, t: end + (start - end) * (1 + math.cos(math.pi * t)) / 2
    if progress < pct:
        return anneal(lr_max / div, lr_max, progress / pct), anneal(0.95, 0.85, progress / pct)
    t = (progress - pct) / (1 - pct)
    return anneal(lr_max, lr_max / div / final_div, t), anneal(0.85, 0.95, t)


def train(minutes, episodes, batch=64, out="paddle_model.pt"):
    torch.manual_seed(0)
    t_data = time.time()
    frames, hidx, nidx, acts, rare = make_data(episodes, seed=0)
    base = 1 + 3 * rare.float()
    prio = base * 0.1
    syms = symmetries()
    model = PaddleWorldModel()
    ema = copy.deepcopy(model).requires_grad_(False)
    print(f"{len(acts)} recorded steps ({int(rare.sum())} hits/misses/serves) in {time.time() - t_data:.0f}s, "
          f"model has {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    t0, s = time.time(), 0
    while (progress := (time.time() - t0) / (60 * minutes)) < 1:
        s += 1
        lr, beta1 = one_cycle(progress)
        for g in opt.param_groups:
            g["lr"], g["betas"] = lr, (beta1, 0.999)
        half = batch // 2
        i = torch.cat([torch.multinomial(prio, half, replacement=True),
                       torch.multinomial(base, batch - half, replacement=True)])
        hist, y, a = frames[hidx[i]].long(), frames[nidx[i]].long(), acts[i].clone()
        which = torch.randint(0, len(syms), (batch,))
        for k, sym in enumerate(syms):
            sel = which == k
            if sel.any():
                hist[sel], y[sel], a[sel] = transform(hist[sel], sym), transform(y[sel], sym), transform_actions(a[sel], sym)
        focus = torch.rand(batch) < 0.5
        ratio = torch.where(focus, torch.rand(batch) ** 3, torch.cos(torch.rand(batch) * math.pi / 2))
        mask = torch.rand(batch, H, W) < ratio[:, None, None]
        mask |= (y != hist[:, -1]) & focus[:, None, None] & (torch.rand(batch, H, W) < 0.9)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(hist, a, torch.where(mask, torch.tensor(MASK), y))
        ce = F.cross_entropy(logits.float(), torch.where(mask, y, torch.tensor(-100)), reduction="none")
        n_masked = mask.sum((1, 2))
        loss = ce.sum() / n_masked.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            prio[i] = base[i] * (ce.sum((1, 2)) / n_masked.clamp(min=1) + 1e-3) ** 0.6
            decay = min(0.999, (1 + s) / (10 + s))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        if s % 250 == 0:
            print(f"step {s}  loss {loss.item():.4f}  lr {lr:.1e}  {time.time() - t0:.0f}s", flush=True)
            torch.save(ema.state_dict(), out)
    torch.save(ema.state_dict(), out)
    print(f"done: {s} steps in {time.time() - t0:.0f}s, saved {out}", flush=True)
    return ema


# ---------- evaluation ----------

def dream(model, seed, steps, eps=0.05):
    """Two bots play `steps` steps watching only the model's imagined frames. The real game
    runs alongside on the same keys; when it serves, it uses the serve the model imagined
    (any valid serve is a legal outcome). Returns frame pairs, first rule break (None = never),
    points scored, passes."""
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    game = PaddleGame(rng)
    styles = (0, int(rng.integers(0, 2)))
    hist = [game.frame()] * K
    pairs, broke, passes = [(hist[-1], hist[-1])], None, 0
    for t in range(steps):
        a = (bot(hist[-1], 0, rng, styles[0], eps), bot(hist[-1], 1, rng, styles[1], eps))
        pred, p = model.generate(torch.from_numpy(np.stack(hist[-K:])), a, gen)
        pred = pred.numpy()
        passes += p
        serve_to = None
        if game.ball[1] in (0, W - 1):
            balls = np.argwhere(is_ball(pred))
            if len(balls) == 1:
                r, c = (int(v) for v in balls[0])
                serve_to = (r, c, *BALL_VEL[int(pred[r, c])])
        real = game.step(*a, serve_to=serve_to)
        pairs.append((real, pred))
        if broke is None and not np.array_equal(real, pred):
            broke = t + 1
        hist.append(pred)
    return pairs, broke, sum(game.score), passes


def evaluate(model, n=100, steps=200, first_seed=50_000):
    clean, until, points, passes = 0, [], 0, 0
    for seed in range(first_seed, first_seed + n):
        pairs, broke, pts, p = dream(model, seed, steps)
        clean += broke is None
        until.append((broke - 1) if broke else steps)
        points, passes = points + pts, passes + p
    res = {"clean": clean, "n": n, "steps": steps, "steps_before_mistake": float(np.mean(until)),
           "points": points, "passes_per_frame": passes / (n * steps)}
    print(f"{clean}/{n} dreams of {steps} steps followed every rule; "
          f"{res['steps_before_mistake']:.0f} steps on average before the first mistake; "
          f"{points} points scored; {res['passes_per_frame']:.2f} passes per frame", flush=True)
    return res


def make_gif(model, path="paddle_dream.gif", tries=20, steps=240):
    best = None
    for seed in range(60_000, 60_000 + tries):
        pairs, broke, pts, _ = dream(model, seed, steps)
        score = (broke or steps + 1) + 20 * pts
        if best is None or score > best[0]:
            best = (score, seed, pairs, broke, pts)
    _, seed, pairs, broke, pts = best
    f16, f13 = font(16), font(13)
    imgs = []
    for t, (real, pred) in enumerate(pairs):
        L, R = render(real, 12), render(pred, 12)
        w, h = L.size
        out = Image.new("RGB", (2 * w + 60, h + 72), (10, 12, 16))
        d = ImageDraw.Draw(out)
        out.paste(L, (20, 40))
        out.paste(R, (w + 40, 40))
        d.text((20, 12), "Real game rules", fill="white", font=f16)
        d.text((w + 40, 12), "World model (imagined)", fill="white", font=f16)
        match = np.array_equal(real, pred)
        if not match:
            d.rectangle([w + 37, 37, 2 * w + 42, h + 42], outline=(255, 70, 70), width=3)
        d.text((20, h + 50), f"step {t}", fill=(170, 170, 170), font=f13)
        d.text((w + 40, h + 50), "matches the real game" if match else "differs from the real game",
               fill=(120, 200, 120) if match else (255, 110, 110), font=f13)
        imgs.append(out)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=90, loop=0)
    print(f"saved {path}: seed {seed}, {len(pairs) - 1} steps, {pts} points, "
          f"{'no mistakes' if broke is None else f'first mistake at step {broke}'}", flush=True)


if __name__ == "__main__":
    arg = lambda name, default, cast=float: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    out = arg("--out", "paddle_model.pt", str)
    if sys.argv[1] == "test":
        test_symmetries()
    elif sys.argv[1] == "train":
        train(minutes=arg("--minutes", 50), episodes=arg("--episodes", 4000, int), out=out)
    else:
        model = PaddleWorldModel()
        model.load_state_dict(torch.load(out))
        model.eval()
        res = evaluate(model, n=arg("--n", 100, int))
        json.dump(res, open(out.replace(".pt", "_eval.json"), "w"))
        make_gif(model)
