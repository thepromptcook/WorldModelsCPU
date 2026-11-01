#!/usr/bin/env bash
# One-time setup: a virtualenv with CPU-only PyTorch (no GPU needed) and the other dependencies.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -c "import torch, chess; print('torch', torch.__version__, '| python-chess', chess.__version__, '| threads', torch.get_num_threads())"
