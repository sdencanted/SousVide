#!/usr/bin/env bash
set -euo pipefail

# Pixi installs the environment before running this task, so Torch is available
# when tiny-cuda-nn builds its PyTorch bindings.
python -c "import torch; print(f'Installing tiny-cuda-nn against torch {torch.__version__}')"
python -m pip install --no-build-isolation \
  'git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch'
