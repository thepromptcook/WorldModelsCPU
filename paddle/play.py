"""Play the paddle game inside the world model: after the first frame, every frame is imagined.

Keys per player: u = up, d = down, . = stay. A side you don't give keys for is played by a bot
that watches the imagined frames.

    python play.py --left uuuu....dddd                 # you (left) vs the bot (right)
    python play.py --left uu..dd --right ..dduu        # two players
    python play.py --steps 120                         # bot vs bot
Saves play.gif and play_last.png.
"""
import sys
import numpy as np
import torch
from PIL import Image
from game import PaddleGame, bot, render
from world_model import K, PaddleWorldModel

KEYS = {".": 0, "u": 1, "d": 2}
arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
left, right = arg("--left", None), arg("--right", None)
seed = int(arg("--seed", 1))
steps = int(arg("--steps", max(len(left or ""), len(right or ""), 1)))

model = PaddleWorldModel()
model.load_state_dict(torch.load(arg("--model", "paddle_model.pt")))
model.eval()
rng, gen = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
history = [PaddleGame(rng).frame()] * K
for t in range(steps):
    a_left = KEYS[left[t]] if left and t < len(left) else bot(history[-1], 0, rng, 0, 0.05)
    a_right = KEYS[right[t]] if right and t < len(right) else bot(history[-1], 1, rng, 0, 0.05)
    frame, _ = model.generate(torch.from_numpy(np.stack(history[-K:])), (a_left, a_right), gen)
    history.append(frame.numpy())
imgs = [render(f) for f in history[K - 1:]]
imgs[0].save("play.gif", save_all=True, append_images=imgs[1:], duration=90, loop=0)
imgs[-1].save("play_last.png")
print(f"saved play.gif ({steps} imagined frames) and play_last.png")
