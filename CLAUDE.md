# Notes for Claude

Small world models (paddle game, chess rules) trained from scratch on a CPU. No GPU anywhere.

- Setup once per container: `./setup.sh` (creates `.venv`; CPU PyTorch from the PyTorch CPU index).
- Each scenario folder is self-contained: `game.py` (real rules, players, drawing) and
  `world_model.py` (model, generation, data, training, evaluation). Run scripts from inside the
  folder with `../.venv/bin/python`.
- Quick checks before committing: `world_model.py test` in each folder (augmentation exactness),
  plus a short smoke run (`train --minutes 0.2` with a few episodes/games, then the `eval` functions).
- Long runs: start them in the background with `THREADS=n` and `--minutes M` (a time budget, so the
  learning-rate schedule finishes on time). Two runs in parallel each get half the CPU. Keep
  `THREADS` × runs ≤ cores.
- Cloud containers can restart during long waits and kill running jobs. Training is resumable:
  re-run the same `train` command and it continues from `<out>_resume.pt` (deleted when a run
  finishes). Start long runs as tracked background tasks and check the log after any restart.
- bfloat16 autocast is used for training; check `grep amx /proc/cpuinfo` on a new machine (without
  AMX/AVX512-BF16 it can be slower than float32).
- Don't commit `*.pt`, GIFs or PNGs outside `results/`.
- Lessons that apply to any new game: put hidden state on screen; generate MaskGIT-style for
  random events; give the network whole-board context; oversample rare events; evaluate by
  rule-checking imagined games against the real game.
