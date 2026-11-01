"""A two-player paddle game (Pong-style) on a 32x32 grid: the real game the world model learns from.

Cells: floor, left paddle, right paddle, or the ball. The ball is drawn with its direction
built in (6 kinds: moving left/right x up/level/down), so one frame shows everything needed
to predict the next.
Each player: 0 stay, 1 up, 2 down. Paddles are 4 cells tall and move 1 cell per step.
The ball moves 1 column per step (and 0 or 1 row). It bounces off the top and bottom walls
and off paddles; hitting a paddle's top or bottom end sends it off at an angle. If a player
misses, the ball is shown in their column for one frame (a point for the other player), then
a new serve appears near the middle at a random row, in a random direction.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, P = 32, 32, 4
FLOOR, LEFT, RIGHT = 0, 1, 2
VELS = [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)]   # (rows, cols) per step
BALL = {v: 3 + i for i, v in enumerate(VELS)}
BALL_VEL = {k: v for v, k in BALL.items()}
N_CLASSES = 9
N_ACTIONS = 3          # per player: stay, up, down
MOVE = {0: 0, 1: -1, 2: 1}
SERVE_ROWS = range(4, H - 4)
SERVE_COLS = (W // 2 - 1, W // 2)


def is_ball(v):
    return v >= 3


class PaddleGame:
    def __init__(self, rng):
        self.rng = rng
        self.left = int(rng.integers(0, H - P + 1))
        self.right = int(rng.integers(0, H - P + 1))
        self.score = [0, 0]
        self.serve()

    def serve(self, to=None):
        if to is not None and to[0] in SERVE_ROWS and to[1] in SERVE_COLS and tuple(to[2:]) in BALL:
            self.ball, self.vel = (int(to[0]), int(to[1])), (int(to[2]), int(to[3]))
        else:
            self.ball = (int(self.rng.choice(SERVE_ROWS)), int(self.rng.choice(SERVE_COLS)))
            self.vel = VELS[int(self.rng.integers(len(VELS)))]

    def frame(self):
        f = np.zeros((H, W), np.int64)
        f[self.left:self.left + P, 0] = LEFT
        f[self.right:self.right + P, W - 1] = RIGHT
        f[self.ball] = BALL[self.vel]
        return f

    def step(self, a_left, a_right, serve_to=None):
        """serve_to=(row, col, dr, dc) optionally fixes a random serve (used to follow a dream)."""
        self.left = min(max(self.left + MOVE[int(a_left)], 0), H - P)
        self.right = min(max(self.right + MOVE[int(a_right)], 0), H - P)
        (r, c), (dr, dc) = self.ball, self.vel
        if c in (0, W - 1):                       # missed last step: point, then a new serve
            self.score[1 if c == 0 else 0] += 1
            self.serve(serve_to)
            return self.frame()
        nr = r + dr
        if not 0 <= nr < H:                       # top/bottom wall
            dr = -dr
            nr = r + dr
        nc = c + dc
        if nc in (0, W - 1):
            top = self.left if nc == 0 else self.right
            if top <= nr < top + P:               # paddle hit: bounce back; the ends give an angle
                off = nr - top
                dr = -1 if off == 0 else 1 if off == P - 1 else dr
                dc = -dc
                nc = c
        self.ball, self.vel = (nr, nc), (dr, dc)
        return self.frame()


def read(frame):
    """What a player can see: paddle tops and the ball (position and direction)."""
    lp, rp = np.where(frame[:, 0] == LEFT)[0], np.where(frame[:, W - 1] == RIGHT)[0]
    balls = np.argwhere(is_ball(frame))
    ball = (tuple(int(v) for v in balls[0]), BALL_VEL[int(frame[tuple(balls[0])])]) if len(balls) == 1 else None
    return (int(lp[0]) if len(lp) else None), (int(rp[0]) if len(rp) else None), ball


def intercept(ball, col):
    """Row where the ball will cross column `col`, following wall bounces."""
    (r, c), (dr, dc) = ball
    steps = (col - c) * dc
    if steps < 0:
        return None
    y = r + dr * steps
    period = 2 * (H - 1)
    y %= period
    return y if y < H else period - y


def bot(frame, side, rng, style, eps):
    """A player that only looks at the screen. Styles: 0 predicts where the ball will arrive,
    1 follows the ball's row, 2 lazy (rarely moves), 3 random. eps = chance of a random key."""
    if style == 3 or rng.random() < eps:
        return int(rng.integers(0, N_ACTIONS))
    lp, rp, ball = read(frame)
    top = lp if side == 0 else rp
    if top is None or ball is None:
        return 0
    if style == 2 and rng.random() < 0.8:
        return 0
    col = 0 if side == 0 else W - 1
    target = intercept(ball, col) if style == 0 else ball[0][0]
    if target is None:
        target = (H - 1) / 2                      # ball going away: drift to the middle
    centre = top + (P - 1) / 2
    if abs(target - centre) <= 1:
        return 0
    return 1 if target < centre else 2


COLORS = {LEFT: (90, 170, 255), RIGHT: (255, 160, 70)}


def render(frame, cell=16):
    img = Image.new("RGB", (W * cell, H * cell), (18, 21, 28))
    d = ImageDraw.Draw(img)
    for r in range(0, H, 2):                       # centre line (decoration)
        d.rectangle([W * cell // 2 - 1, r * cell + 3, W * cell // 2 + 1, r * cell + cell - 4], fill=(45, 50, 62))
    for (r, c), v in np.ndenumerate(frame):
        x, y = c * cell, r * cell
        if v in (LEFT, RIGHT):
            d.rectangle([x + 3, y, x + cell - 4, y + cell - 1], fill=COLORS[int(v)])
        elif is_ball(v):
            dr, dc = BALL_VEL[int(v)]              # short trail behind the ball shows its direction
            cx, cy = x + cell // 2, y + cell // 2
            d.line([cx, cy, cx - dc * cell * 0.8, cy - dr * cell * 0.8], fill=(110, 110, 120), width=4)
            d.ellipse([x + 2, y + 2, x + cell - 3, y + cell - 3], fill=(245, 245, 245))
    return img


def font(size=15):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()
