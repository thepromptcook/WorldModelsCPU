# World models on a CPU

Small, complete **world models** you can create and train from scratch in a CPU-only container
(built and tested in a Claude Code cloud session: 4 cores, no GPU). A world model learns the
rules of a game purely by watching it played, then *imagines* the game: given the current
frame and the players' inputs, it generates the next frame. After the first frame, no game
code runs.

| Scenario | Grid | Inputs per step | What makes it hard |
|---|---|---|---|
| [`paddle/`](paddle/): a two-player Pong-style game | 32×32 cells | both players' keys (3×3 combinations) | ball physics (bounces, paddle angles), two players, random serves |
| [`chess_wm/`](chess_wm/): the rules of chess | 9×8 tokens (board + status row) | a move attempt (from, to, promotion) | sliding pieces, pins, check/mate, castling, en passant, illegal moves |

## Setup and running

```bash
./setup.sh                                   # once: .venv with CPU PyTorch, numpy, pillow, python-chess
cd paddle
THREADS=4 ../.venv/bin/python world_model.py train --minutes 50 --episodes 4000
../.venv/bin/python world_model.py eval      # rule-checked imagined games + paddle_dream.gif
../.venv/bin/python play.py --left uuuu....dddd   # play inside the model (bot plays the other side)
cd ../chess_wm
THREADS=4 ../.venv/bin/python world_model.py train --minutes 40 --games 4000
../.venv/bin/python world_model.py eval      # accuracy by move kind + imagined games + chess_dream.gif
```

`THREADS` sets the CPU threads. Each `world_model.py test` checks that the data augmentation is
an exact symmetry of the real game.

## How it works

1. **The real game** (`game.py`) produces frames as a grid of tokens, one per cell or square.
   Every frame shows everything needed to predict the next one: the ball is drawn with its
   direction, and chess has a status row for turn, castling rights and the en-passant file.
2. **Recorded play:** scripted players with varied styles and skill (including careless ones
   and random key presses) generate a million or so recorded steps in a couple of minutes.
3. **The world model** takes the current frame and the inputs, and predicts the next frame. The
   paddle model is a small convolutional network with whole-board context (average + max pooling
   into an MLP). The chess model is a small transformer, because chess rules reach across the board.
4. **Generation, Genie/MaskGIT-style:** start with every cell hidden, predict all of them, keep
   the ones the model is sure about, and predict the rest again with those filled in. The last
   few uncertain cells go one at a time. That lets the model *choose* an outcome for random
   events (where a serve goes) instead of averaging them into nothing.
5. **Evaluation:** players play *inside the model's imagination* while the real game runs
   alongside on the same inputs. Random outcomes the model imagined are copied into the real
   game, because any legal outcome counts. A run passes only if every frame matches exactly.

## Training recipe (all CPU)

- **Hide-and-predict training:** half the batch hides a random fraction of the next frame;
  the other half mostly hides the cells that changed. That practises the "nearly finished,
  where does the random event go?" situations that generation ends in.
- **Rare-event boosting:** hits, misses, serves (paddle) and captures, castling, en passant,
  promotions, checks, mates (chess) are shown 4× as often.
- **Exact symmetry augmentation:** paddle has 4 (mirroring left/right also swaps the players);
  chess has 1 (flip the board and swap colours). Both are verified against the real game.
- **Prioritized replay:** half of each batch is drawn by how badly the model did last time.
- **Weight averaging (EMA)** of the saved model, **bfloat16** on the CPU's AMX units (about 2×
  faster on recent Xeons), and a **time budget** with a one-cycle learning-rate schedule.

## Lessons from building these

- **The screen must contain the answer.** If a frame doesn't show something (which end of a
  coiled snake is the tail, which way a ball is moving, castling rights), the model can't
  predict it reliably. Put hidden state on screen, or give the model memory.
- **Randomness needs generation, not prediction.** Predicting the single most likely frame
  makes random events (new food, a serve) vanish. Cell-by-cell sampling fixes it.
- **Whole-board rules need whole-board reasoning.** "Has the random object appeared anywhere
  yet?" needs max pooling or attention; averaging hides a single cell.
- **Rare moments decide accuracy.** Most mistakes happen at rare events, so oversample them.

## Results

Filled in from the runs in this container (see `results/`).
