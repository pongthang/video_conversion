#!/usr/bin/env bash
# One-shot setup: creates a local virtualenv, installs every dependency and
# downloads the models into ./models. Safe to re-run.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN=""
FORCE_CPU=0
WITH_DEMUCS=0
WITH_KOKORO=0
WITH_GUI=1
SKIP_MODELS=0
ASR_MODEL=""
PIPER_VOICE=""
TORCH_VERSION="2.5.1"
CUDA_TAG="cu121"

RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; BLUE=$'\033[34m'; BOLD=$'\033[1m'; OFF=$'\033[0m'
info()  { printf '%s==>%s %s\n' "$BLUE$BOLD" "$OFF" "$*"; }
ok()    { printf '%s  ok%s %s\n' "$GREEN" "$OFF" "$*"; }
warn()  { printf '%swarn%s %s\n' "$YELLOW" "$OFF" "$*" >&2; }
die()   { printf '%s err%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./setup.sh [options]

  --python PATH      Python interpreter to build the venv from (3.9-3.11)
  --cpu              Install the CPU-only build of PyTorch
  --with-demucs      Also install Demucs (keeps original music under the dub)
  --with-kokoro      Also install the Kokoro TTS voice (better prosody)
  --with-gui         Install the desktop application (default)
  --no-gui           Command line only; skip the desktop interface
  --asr-model NAME   Whisper model to pre-download (default: from config)
  --voice NAME       Piper voice to pre-download (default: from config)
  --skip-models      Install packages only, download models on first run
  -h, --help         Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)      PYTHON_BIN="$2"; shift 2 ;;
    --cpu)         FORCE_CPU=1; shift ;;
    --with-demucs) WITH_DEMUCS=1; shift ;;
    --with-kokoro) WITH_KOKORO=1; shift ;;
    --with-gui)    WITH_GUI=1; shift ;;
    --no-gui)      WITH_GUI=0; shift ;;
    --asr-model)   ASR_MODEL="$2"; shift 2 ;;
    --voice)       PIPER_VOICE="$2"; shift 2 ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             die "Unknown option: $1 (try --help)" ;;
  esac
done

# ---------------------------------------------------------------- system deps
info "Checking system tools"
for tool in ffmpeg ffprobe; do
  command -v "$tool" >/dev/null 2>&1 || die "$tool not found. Install it first:
    Debian/Ubuntu: sudo apt install ffmpeg
    Fedora:        sudo dnf install ffmpeg
    macOS:         brew install ffmpeg"
done
FFMPEG_VERSION="$(ffmpeg -version 2>/dev/null | sed -n '1s/^ffmpeg version \([^ ]*\).*/\1/p')"
ok "ffmpeg ${FFMPEG_VERSION:-installed}"

# Note: no pipe into `grep -q` here - under `set -o pipefail` grep's early exit
# would SIGPIPE ffmpeg and make the whole test look like a failure.
FFMPEG_FILTERS="$(ffmpeg -hide_banner -filters 2>/dev/null || true)"
if [[ "$FFMPEG_FILTERS" == *" rubberband "* ]]; then
  ok "rubberband filter available (best quality time stretching)"
else
  warn "ffmpeg has no rubberband filter; falling back to atempo (slightly lower quality)"
fi

if command -v espeak-ng >/dev/null 2>&1; then
  ok "espeak-ng available"
else
  warn "espeak-ng not found. Piper works without it, Kokoro needs it for unusual words:
       sudo apt install espeak-ng"
fi

# --------------------------------------------------------------- interpreter
# A candidate counts only if it actually runs and can build a venv: pyenv
# leaves shims on PATH for versions that are not selected, and some distro
# pythons ship without ensurepip.
python_ok() {
  local candidate="$1" version major minor
  version="$("$candidate" -c 'import ensurepip, sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)" || return 1
  [[ -n "$version" ]] || return 1
  major="${version%%.*}"; minor="${version##*.}"
  [[ "$major" -eq 3 && "$minor" -ge 9 && "$minor" -le 11 ]]
}

find_python() {
  if [[ -n "$PYTHON_BIN" ]]; then echo "$PYTHON_BIN"; return; fi
  local candidates=()
  # Real pyenv interpreters first: 3.11 has wheels for every dependency.
  if command -v pyenv >/dev/null 2>&1; then
    local root; root="$(pyenv root 2>/dev/null || true)"
    if [[ -n "$root" ]]; then
      for series in 3.11 3.10 3.9; do
        for p in "$root"/versions/"$series".*/bin/python3; do
          [[ -x "$p" ]] && candidates+=("$p")
        done
      done
    fi
  fi
  candidates+=(python3.11 python3.10 python3.9 python3)
  for candidate in "${candidates[@]}"; do
    command -v "$candidate" >/dev/null 2>&1 || continue
    if python_ok "$candidate"; then echo "$candidate"; return; fi
  done
  echo ""
}

info "Locating a suitable Python (3.9-3.11)"
PY="$(find_python)"
[[ -n "$PY" ]] || die "No usable Python 3.9-3.11 found (it must also provide venv/ensurepip).
    Debian/Ubuntu: sudo apt install python3.11 python3.11-venv
    pyenv:         pyenv install 3.11.13
    Or point at one directly: ./setup.sh --python /path/to/python3.11"
ok "$("$PY" -c 'import sys; print(sys.executable, "-", sys.version.split()[0])')"

# --------------------------------------------------------------------- venv
if [[ -d "$VENV_DIR" ]]; then
  info "Reusing existing virtualenv at .venv"
else
  info "Creating virtualenv at .venv"
  "$PY" -m venv "$VENV_DIR" || die "venv creation failed. On Debian/Ubuntu you may need:
    sudo apt install python3-venv"
fi
VPY="$VENV_DIR/bin/python"
"$VPY" -m pip install --quiet --upgrade pip setuptools wheel
ok "pip $("$VPY" -m pip --version | awk '{print $2}')"

# -------------------------------------------------------------------- torch
detect_cuda() {
  [[ "$FORCE_CPU" -eq 1 ]] && return 1
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

if "$VPY" -c 'import torch' >/dev/null 2>&1; then
  info "PyTorch already installed ($("$VPY" -c 'import torch; print(torch.__version__)'))"
else
  if detect_cuda; then
    info "NVIDIA GPU detected - installing PyTorch $TORCH_VERSION ($CUDA_TAG)"
    "$VPY" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" \
      --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
      || die "PyTorch install failed. Retry with ./setup.sh --cpu for a CPU-only build."
  else
    info "No GPU detected - installing the CPU build of PyTorch"
    "$VPY" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCH_VERSION" \
      --index-url "https://download.pytorch.org/whl/cpu"
  fi
fi

# --------------------------------------------------------------- python deps
info "Installing pipeline dependencies"
"$VPY" -m pip install -r requirements.txt

if [[ "$WITH_GUI" -eq 1 ]]; then
  info "Installing the desktop interface (PySide6)"
  # Essentials rather than the full PySide6: it is a third of the size and
  # carries every module the application actually imports.
  "$VPY" -m pip install "PySide6-Essentials==6.8.1"
fi

if [[ "$WITH_DEMUCS" -eq 1 || "$WITH_KOKORO" -eq 1 ]]; then
  info "Installing optional components"
  [[ "$WITH_DEMUCS" -eq 1 ]] && "$VPY" -m pip install "demucs==4.0.1"
  [[ "$WITH_KOKORO" -eq 1 ]] && "$VPY" -m pip install "kokoro>=0.9.2" "misaki[en]>=0.9.3"
fi

info "Installing the vtrans package (editable)"
"$VPY" -m pip install --no-deps -e .

# --------------------------------------------------------------- sanity check
info "Verifying the install"
VTRANS_CHECK_GUI="$WITH_GUI" "$VPY" - <<'PYCHECK'
import sys
problems = []
try:
    import torch
    print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  gpu   {torch.cuda.get_device_name(0)}")
except Exception as exc:
    problems.append(f"torch: {exc}")
checks = [("faster_whisper", "faster-whisper"), ("transformers", "transformers"),
          ("piper.voice", "piper-tts"), ("soundfile", "soundfile"),
          ("yaml", "PyYAML"), ("vtrans", "vtrans")]
import os
if os.environ.get("VTRANS_CHECK_GUI") == "1":
    checks += [("PySide6.QtWidgets", "PySide6"), ("vtrans_gui", "vtrans_gui")]
for module, label in checks:
    try:
        __import__(module)
        print(f"  ok    {label}")
    except Exception as exc:
        problems.append(f"{label}: {exc}")
if problems:
    print("\nFAILED:", file=sys.stderr)
    for p in problems:
        print("  " + p, file=sys.stderr)
    sys.exit(1)
PYCHECK

# ------------------------------------------------------------------- models
if [[ "$SKIP_MODELS" -eq 0 ]]; then
  info "Downloading models into ./models (a few GB, only happens once)"
  DL_ARGS=()
  [[ -n "$ASR_MODEL" ]]   && DL_ARGS+=(--asr-model "$ASR_MODEL")
  [[ -n "$PIPER_VOICE" ]] && DL_ARGS+=(--piper-voice "$PIPER_VOICE")
  [[ "$WITH_KOKORO" -eq 1 ]] && DL_ARGS+=(--kokoro)
  [[ "$WITH_DEMUCS" -eq 1 ]] && DL_ARGS+=(--demucs)
  "$VPY" -m vtrans.download "${DL_ARGS[@]}"
else
  warn "Skipping model download; the first conversion will fetch them."
fi

chmod +x "$PROJECT_DIR/convert_video.sh" 2>/dev/null || true

chmod +x "$PROJECT_DIR/run_gui.sh" "$PROJECT_DIR/install_desktop.sh" 2>/dev/null || true

GUI_LINES=""
if [[ "$WITH_GUI" -eq 1 ]]; then
  GUI_LINES="
  Open the desktop app:
    ./run_gui.sh

  Add it to the applications menu:
    ./install_desktop.sh
"
fi

cat <<DONE

${GREEN}${BOLD}Setup complete.${OFF}
${GUI_LINES}
  Convert a video from the command line:
    ./convert_video.sh input.mp4 -out output.mp4

  Convert a YouTube link:
    ./convert_video.sh "https://www.youtube.com/watch?v=..." -out output.mp4

  All options:
    ./convert_video.sh --help
DONE
