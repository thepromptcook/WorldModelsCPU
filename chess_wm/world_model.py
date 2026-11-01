"""A world model of the rules of chess: (position tokens, move) -> next position tokens.

A small transformer over the 72 tokens (chess rules reach across the board: sliding pieces,
pins, checks), generating MaskGIT-style like the paddle model. Trained with: rare-move
boosting (castling, en passant, promotion, checks, mates), colour-flip augmentation (mirror the
board and swap colours: an exact symmetry of chess), prioritized replay, weight averaging (EMA),
bfloat16 and a time budget.

    THREADS=4 python world_model.py train --minutes 40 --games 4000
    python world_model.py eval        # move accuracy by kind + rule-checked games + chess_dream.gif
    python world_model.py test        # check the colour-flip augmentation against python-chess
"""
import copy
import json
import math
import os
import sys
import time
from multiprocessing import Pool
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from game import (N_TOKENS, N_CLASSES, PROMOS, tokens, board_from_tokens, step, encode_move, move_kind,
                  choose_move, play_game, render, game_over, cell_index, _font)

MASK = N_CLASSES
torch.set_num_threads(int(os.environ.get("THREADS", 4)))


class ChessWorldModel(nn.Module):
    def __init__(self, d=128, layers=6, heads=4):
        super().__init__()
        self.cur = nn.Embedding(N_CLASSES, d)
        self.part = nn.Embedding(N_CLASSES + 1, d)
        self.pos = nn.Parameter(torch.randn(N_TOKENS, d) * 0.02)
        self.src, self.dst, self.promo = nn.Embedding(2, d), nn.Embedding(2, d), nn.Embedding(len(PROMOS), d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, N_CLASSES))

    def forward(self, cur, action, partial):
        """cur, partial (B,72); action (B,3) = from token index, to token index, promotion."""
        idx = torch.arange(N_TOKENS)
        f = (idx[None] == action[:, :1]).long()
        t = (idx[None] == action[:, 1:2]).long()
        x = self.cur(cur) + self.part(partial) + self.pos + self.src(f) + self.dst(t) + self.promo(action[:, 2])[:, None]
        return self.head(self.enc(x))

    @torch.no_grad()
    def generate(self, cur, action, gen=None, top_p=0.95, commit_above=0.999):
        """Next position, MaskGIT-style. cur (72,), action (3,). Returns (tokens, passes)."""
        out = torch.full((N_TOKENS,), MASK, dtype=torch.long)
        masked = torch.ones(N_TOKENS, dtype=torch.bool)
        passes = 0
        while masked.any():
            passes += 1
            probs = self(cur[None], action[None], out[None])[0].softmax(-1)
            sp, si = probs.sort(-1, descending=True)
            filtered = torch.zeros_like(probs).scatter(-1, si, sp * ((sp.cumsum(-1) - sp) < top_p))
            sample = torch.multinomial(filtered, 1, generator=gen)[:, 0]
            conf = probs.gather(-1, sample[:, None])[:, 0]
            conf[~masked] = -1
            m = int(masked.sum())
            keep = conf.topk(max(int((conf > commit_above).sum()), (m + 1) // 2 if m > 8 else 1)).indices
            out[keep], masked[keep] = sample[keep], False
        return out, passes


# ---------- data ----------

def make_data(n_games, seed, workers):
    seeds = [seed * 1_000_000 + g for g in range(n_games)]
    if workers > 1:
        with Pool(workers) as pool:
            games = pool.map(play_game, seeds, chunksize=16)
    else:
        games = [play_game(s) for s in seeds]
    frames, cur_idx, acts, special, total = [], [], [], [], 0
    for f, a, s in games:
        frames.append(f)
        cur_idx.append(total + np.arange(len(a)))
        acts.append(a)
        special.append(s)
        total += len(f)
    cat = lambda xs: torch.from_numpy(np.concatenate(xs))
    return cat(frames), cat(cur_idx), cat(acts), cat(special)


# ---------- colour-flip augmentation ----------

FLIP_CLASS = torch.arange(N_CLASSES)
FLIP_CLASS[1:7], FLIP_CLASS[7:13] = torch.arange(7, 13), torch.arange(1, 7)
FLIP_CLASS[13], FLIP_CLASS[14] = 14, 13
FLIP_INDEX = torch.cat([torch.arange(64).view(8, 8).flip(0).reshape(64),
                        torch.tensor([64, 67, 68, 65, 66, 69, 70, 71])])   # swap white/black castling cells


def flip_tokens(x):
    """Mirror the board top-to-bottom and swap colours (python-chess Board.mirror())."""
    return FLIP_CLASS[x[..., FLIP_INDEX]]


def flip_action(a):
    a = a.clone()
    a[..., :2] = FLIP_INDEX[a[..., :2]]
    return a


def test_flip(games=200):
    rng = np.random.default_rng(1)
    for g in range(games):
        b = chess.Board()
        for _ in range(150):
            t = torch.from_numpy(tokens(b))
            assert torch.equal(flip_tokens(t), torch.from_numpy(tokens(b.mirror()))), "position mismatch"
            m = choose_move(b, rng, g % 2, 0.15)
            mm = chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
            assert torch.equal(flip_action(torch.tensor(encode_move(m))), torch.tensor(encode_move(mm)))
            b2 = b.mirror()
            step(b, m)
            step(b2, mm)
            assert torch.equal(flip_tokens(torch.from_numpy(tokens(b))), torch.from_numpy(tokens(b2))), "move mismatch"
            if game_over(b):
                break
    print(f"colour-flip augmentation exact on {games} games")


# ---------- training ----------

def one_cycle(progress, lr_max=1e-3, pct=0.3, div=25, final_div=1e4):
    anneal = lambda start, end, t: end + (start - end) * (1 + math.cos(math.pi * t)) / 2
    if progress < pct:
        return anneal(lr_max / div, lr_max, progress / pct), anneal(0.95, 0.85, progress / pct)
    t = (progress - pct) / (1 - pct)
    return anneal(lr_max, lr_max / div / final_div, t), anneal(0.85, 0.95, t)


def train(minutes, n_games, batch=128, out="chess_model.pt"):
    torch.manual_seed(0)
    t_data = time.time()
    frames, cur_idx, acts, special = make_data(n_games, seed=1, workers=int(os.environ.get("THREADS", 4)))
    base = 1 + 3 * special.float()
    prio = base * 0.1
    model = ChessWorldModel()
    ema = copy.deepcopy(model).requires_grad_(False)
    print(f"{len(acts)} recorded move attempts ({int(special.sum())} special) in {time.time() - t_data:.0f}s, "
          f"model has {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    t0, s = time.time(), 0
    while (progress := (time.time() - t0) / (60 * minutes)) < 1:
        s += 1
        lr, beta1 = one_cycle(progress)
        for g in opt.param_groups:
            g["lr"], g["betas"] = lr, (beta1, 0.999)
        half = batch // 2
        i = torch.cat([torch.multinomial(prio, half, replacement=True),
                       torch.multinomial(base, batch - half, replacement=True)])
        cur, y, a = frames[cur_idx[i]].long(), frames[cur_idx[i] + 1].long(), acts[i]
        flip = torch.rand(batch) < 0.5
        cur[flip], y[flip], a = flip_tokens(cur[flip]), flip_tokens(y[flip]), torch.where(flip[:, None], flip_action(a), a)
        focus = torch.rand(batch) < 0.5
        ratio = torch.where(focus, torch.rand(batch) ** 3, torch.cos(torch.rand(batch) * math.pi / 2))
        mask = torch.rand(batch, N_TOKENS) < ratio[:, None]
        mask |= (y != cur) & focus[:, None] & (torch.rand(batch, N_TOKENS) < 0.9)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits = model(cur, a, torch.where(mask, torch.tensor(MASK), y))
        ce = F.cross_entropy(logits.float().transpose(1, 2), torch.where(mask, y, torch.tensor(-100)), reduction="none")
        n_masked = mask.sum(1)
        loss = ce.sum() / n_masked.sum().clamp(min=1)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            prio[i] = base[i] * (ce.sum(1) / n_masked.clamp(min=1) + 1e-3) ** 0.6
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

def move_accuracy(model, n_games=150, seed=7):
    """Feed real positions; how often is the predicted next position exactly right, by move kind?"""
    stats, gen = {}, torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    for g in range(n_games):
        board = chess.Board()
        for _ in range(200):
            m = choose_move(board, rng, g % 2, 0.15)
            kind = move_kind(board, m)
            cur = torch.from_numpy(tokens(board))
            pred, _ = model.generate(cur, torch.tensor(encode_move(m)), gen)
            step(board, m)
            ok, n = stats.get(kind, (0, 0))
            stats[kind] = (ok + int(np.array_equal(pred.numpy(), tokens(board))), n + 1)
            if game_over(board):
                break
    return stats


def dream(model, seed, plies=100):
    """Two players pick moves by looking only at the model's imagined positions; the real board
    follows the same move attempts. Returns frames, first mismatch (None = never)."""
    rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    board, styles = chess.Board(), (1, int(rng.integers(0, 2)))
    cur = torch.from_numpy(tokens(board))
    pairs, broke = [(tokens(board), cur.numpy(), None)], None
    for ply in range(plies):
        m = choose_move(board_from_tokens(cur.numpy()), rng, styles[ply % 2], 0.1)
        a = encode_move(m)
        cur, _ = model.generate(cur, torch.tensor(a), gen)
        step(board, m)
        real = tokens(board)
        pairs.append((real, cur.numpy(), a))
        if broke is None and not np.array_equal(real, cur.numpy()):
            broke = ply + 1
        if game_over(board) and game_over(board_from_tokens(cur.numpy())):
            break
    return pairs, broke


def evaluate(model, n_dreams=100):
    stats = move_accuracy(model)
    print("single-move accuracy (real position in, next position exactly right):")
    for kind, (ok, n) in sorted(stats.items(), key=lambda kv: -kv[1][1]):
        print(f"  {kind:16} {100 * ok / n:6.2f}%  ({ok}/{n})")
    clean, until = 0, []
    for seed in range(70_000, 70_000 + n_dreams):
        pairs, broke = dream(model, seed)
        clean += broke is None
        until.append((broke - 1) if broke else len(pairs) - 1)
    print(f"{clean}/{n_dreams} imagined games (up to 100 moves) matched the real rules at every move; "
          f"{np.mean(until):.0f} moves on average before the first mistake", flush=True)
    return {"move_accuracy": {k: [ok, n] for k, (ok, n) in stats.items()}, "dreams_clean": clean,
            "dreams": n_dreams, "moves_before_mistake": float(np.mean(until))}


def make_gif(model, path="chess_dream.gif", tries=20):
    best = None
    for seed in range(80_000, 80_000 + tries):
        pairs, broke = dream(model, seed)
        score = (broke or 1000) + len(pairs)
        if best is None or score > best[0]:
            best = (score, seed, pairs, broke)
    _, seed, pairs, broke = best
    f16, f13 = _font(16), _font(13)
    imgs = []
    for t, (real, pred, a) in enumerate(pairs):
        L, R = render(real, a), render(pred, a)
        w, h = L.size
        out = Image.new("RGB", (2 * w + 60, h + 70), (10, 12, 16))
        d = ImageDraw.Draw(out)
        out.paste(L, (20, 40))
        out.paste(R, (w + 40, 40))
        d.text((20, 12), "Real chess rules", fill="white", font=f16)
        d.text((w + 40, 12), "World model (imagined)", fill="white", font=f16)
        match = np.array_equal(real, pred)
        if not match:
            d.rectangle([w + 37, 37, 2 * w + 42, h + 42], outline=(255, 70, 70), width=3)
        d.text((20, h + 48), f"move {t}", fill=(170, 170, 170), font=f13)
        d.text((w + 40, h + 48), "matches the real rules" if match else "differs from the real rules",
               fill=(120, 200, 120) if match else (255, 110, 110), font=f13)
        imgs.append(out)
    imgs += [imgs[-1]] * 4
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=500, loop=0)
    print(f"saved {path}: seed {seed}, {len(pairs) - 1} moves, "
          f"{'no mistakes' if broke is None else f'first mistake at move {broke}'}", flush=True)


if __name__ == "__main__":
    arg = lambda name, default, cast=float: cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default
    out = arg("--out", "chess_model.pt", str)
    if sys.argv[1] == "test":
        test_flip()
    elif sys.argv[1] == "train":
        train(minutes=arg("--minutes", 40), n_games=arg("--games", 4000, int), out=out)
    else:
        model = ChessWorldModel()
        model.load_state_dict(torch.load(out))
        model.eval()
        res = evaluate(model, n_dreams=arg("--n", 100, int))
        json.dump(res, open(out.replace(".pt", "_eval.json"), "w"), indent=1)
        make_gif(model)
