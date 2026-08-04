#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv_bin="${UV_BIN:-uv}"
colmap_dir="${COLMAP_DIR:-/home/airlab/colmap}"

if [[ -x "$repo_root/.tools/uv/uv" ]]; then
    uv_bin="$repo_root/.tools/uv/uv"
fi

"$uv_bin" venv --python 3.10 "$repo_root/.venv"
"$uv_bin" pip sync --python "$repo_root/.venv/bin/python" "$repo_root/requirements.txt"

# FiGS relies on its vendored Nerfstudio, Hierarchical-Localization, and v2e checkouts.
"$uv_bin" pip install --python "$repo_root/.venv/bin/python" \
    "$repo_root/FiGS/nerfstudio" \
    "$repo_root/FiGS/nerfstudio/third_party/Hierarchical-Localization" \
    "$repo_root/FiGS/v2e"

# tiny-cuda-nn must build against the Torch that the first sync installed.
"$uv_bin" pip install --python "$repo_root/.venv/bin/python" \
    --no-build-isolation --no-deps \
    -r "$repo_root/requirements.tinycudann.txt"

export VIRTUAL_ENV="$repo_root/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
"$colmap_dir/python/build.sh"
