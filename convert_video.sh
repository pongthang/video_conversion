#!/usr/bin/env bash
# Convert a Chinese video into an English-dubbed, English-subtitled video.
#
#   ./convert_video.sh input.mp4 -out output.mp4
#   ./convert_video.sh "https://youtu.be/XXXX" -out output.mp4 --keep-background
#   ./convert_video.sh --help
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VPY="$VENV_DIR/bin/python"

if [[ ! -x "$VPY" ]]; then
  printf '\033[31mError:\033[0m virtualenv not found at %s\n' "$VENV_DIR" >&2
  printf 'Run ./setup.sh first.\n' >&2
  exit 1
fi

# CTranslate2 (faster-whisper) loads cuBLAS and cuDNN at runtime. Those ship as
# pip packages alongside the CUDA build of torch but are not on the loader path,
# so point LD_LIBRARY_PATH at them before importing anything.
NVIDIA_LIBS="$("$VPY" - <<'PY' 2>/dev/null || true
import os, importlib
paths = []
for module in ("nvidia.cublas.lib", "nvidia.cudnn.lib", "nvidia.cuda_runtime.lib"):
    try:
        paths.append(os.path.dirname(importlib.import_module(module).__file__))
    except Exception:
        pass
print(os.pathsep.join(paths))
PY
)"
if [[ -n "$NVIDIA_LIBS" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Put the venv first on PATH so yt-dlp and demucs resolve to the versions this
# project installed, not to older copies that may be on the user's PATH.
export PATH="$VENV_DIR/bin:$PATH"

# Keep every model download inside the project directory.
export HF_HOME="${HF_HOME:-$PROJECT_DIR/models/hf}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/models/torch}"
# Leave a couple of cores for ffmpeg; oversubscribing slows the whole run down.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(( $(nproc) > 2 ? $(nproc) - 2 : 1 ))}"
export TOKENIZERS_PARALLELISM=false

exec "$VPY" -m vtrans "$@"
