#!/usr/bin/env bash
# Launch the desktop application.
#
#   ./run_gui.sh
#
# Mirrors convert_video.sh: same virtualenv, same model locations. The CUDA
# libraries no longer need LD_LIBRARY_PATH set here - vtrans.runtime loads them
# by absolute path at start-up, which is what lets the app work when it is
# started from the desktop menu rather than from a shell.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
VPY="$VENV_DIR/bin/python"

die() {
  # Started from a menu there is no terminal to read, so say it in a dialog too.
  printf '\033[31mError:\033[0m %s\n' "$1" >&2
  if command -v zenity >/dev/null 2>&1; then
    zenity --error --no-wrap --title="Video Translator" --text="$1" 2>/dev/null || true
  elif command -v kdialog >/dev/null 2>&1; then
    kdialog --error "$1" 2>/dev/null || true
  fi
  exit 1
}

[[ -x "$VPY" ]] || die "Virtualenv not found at $VENV_DIR.\n\nRun ./setup.sh first."

if ! "$VPY" -c 'import PySide6' >/dev/null 2>&1; then
  die "The desktop interface is not installed.\n\nRun:  ./setup.sh --with-gui"
fi

export PATH="$VENV_DIR/bin:$PATH"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/models/hf}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/models/torch}"
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT_DIR"
exec "$VPY" -m vtrans_gui "$@"
