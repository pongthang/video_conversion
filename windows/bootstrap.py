"""First-run setup: fetch the heavy pieces the installer deliberately omits.

The installer is small on purpose. PyTorch with CUDA is about 2.5 GB and the
models are another 1-6 GB depending on the tier chosen, so bundling everything
would mean an 8 GB download for every user regardless of what their machine can
actually use. Instead the installer ships the app, a standalone CPython and
this script, which works out what the machine needs and fetches only that.

Run by the installer with a progress window, and again from the Start menu
entry "Repair Video Translator" if anything needs restoring. Every step is
skipped when already satisfied, so re-running is cheap and safe.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, List, Optional

HERE = Path(__file__).resolve().parent
INSTALL_ROOT = HERE.parent if (HERE.name == "windows") else HERE

TORCH_VERSION = "2.5.1"
CUDA_TAG = "cu121"

# A static ffmpeg build, so nothing has to be installed system-wide and the
# uninstaller can really remove everything.
FFMPEG_URL = ("https://github.com/GyanD/codexffmpeg/releases/download/"
              "7.1/ffmpeg-7.1-essentials_build.zip")

PRESET_MODELS = {
    "fast": {"asr": "small", "translate": "facebook/nllb-200-distilled-600M",
             "voices": ["en_US-amy-medium", "en_US-ryan-high"]},
    "balanced": {"asr": "distil-large-v3", "translate": "facebook/nllb-200-distilled-600M",
                 "voices": ["en_US-amy-medium", "en_US-ryan-high"]},
    "best": {"asr": "large-v3", "translate": "facebook/nllb-200-distilled-1.3B",
             "voices": ["en_US-amy-medium", "en_US-ryan-high",
                        "en_US-libritts_r-medium", "en_GB-alan-medium"]},
}


class Reporter:
    """Progress out to whatever is driving this: a GUI, or the console."""

    def __init__(self, as_json: bool = False) -> None:
        self.as_json = as_json

    def step(self, name: str, fraction: float, detail: str = "") -> None:
        if self.as_json:
            print(json.dumps({"t": "step", "name": name, "frac": round(fraction, 4),
                              "detail": detail}), flush=True)
        else:
            bar = "#" * int(fraction * 40)
            print(f"\r[{bar:<40}] {fraction*100:5.1f}%  {name} {detail}",
                  end="", flush=True)

    def log(self, message: str) -> None:
        if self.as_json:
            print(json.dumps({"t": "log", "msg": message}), flush=True)
        else:
            print(f"\n{message}", flush=True)

    def done(self, ok: bool, message: str = "") -> None:
        if self.as_json:
            print(json.dumps({"t": "done", "ok": ok, "msg": message}), flush=True)
        else:
            print(f"\n{'OK' if ok else 'FAILED'}: {message}", flush=True)


def python_exe() -> str:
    """The bundled interpreter, which is the one the app runs under."""
    for candidate in (INSTALL_ROOT / "runtime" / "python.exe",
                      INSTALL_ROOT / "runtime" / "bin" / "python.exe"):
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def has_nvidia_gpu() -> bool:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        proc = subprocess.run([exe, "-L"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=15)
        return proc.returncode == 0 and b"GPU" in proc.stdout
    except (subprocess.SubprocessError, OSError):
        return False


def pip_install(args: List[str], reporter: Reporter, label: str) -> None:
    cmd = [python_exe(), "-m", "pip", "install", "--no-input",
           "--disable-pip-version-check"] + args
    reporter.log(f"Installing {label}...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    tail: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line.rstrip())
        del tail[:-30]
        if line.startswith(("Collecting", "Downloading", "Installing")):
            reporter.log("  " + line.strip()[:110])
    if proc.wait() != 0:
        raise RuntimeError(f"Installing {label} failed:\n" + "\n".join(tail[-15:]))


def download(url: str, dest: Path, reporter: Reporter, label: str,
             base: float, span: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        reporter.log(f"{label} already downloaded")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            read = 0
            with open(tmp, "wb") as handle:
                while True:
                    block = response.read(1 << 18)
                    if not block:
                        break
                    handle.write(block)
                    read += len(block)
                    if total:
                        reporter.step(label, base + span * (read / total),
                                      f"{read/1e6:.0f} / {total/1e6:.0f} MB")
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(f"Could not download {label}: {exc}") from exc
    tmp.replace(dest)
    return dest


def install_ffmpeg(reporter: Reporter, base: float, span: float) -> None:
    bin_dir = INSTALL_ROOT / "bin"
    if (bin_dir / "ffmpeg.exe").is_file() and (bin_dir / "ffprobe.exe").is_file():
        reporter.log("ffmpeg already present")
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = download(FFMPEG_URL, Path(tmp) / "ffmpeg.zip", reporter,
                           "ffmpeg", base, span * 0.8)
        reporter.step("ffmpeg", base + span * 0.85, "extracting")
        bin_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                name = Path(member).name
                if name in ("ffmpeg.exe", "ffprobe.exe", "ffplay.exe"):
                    with zf.open(member) as src, open(bin_dir / name, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    if not (bin_dir / "ffmpeg.exe").is_file():
        raise RuntimeError("The ffmpeg archive did not contain ffmpeg.exe")
    reporter.step("ffmpeg", base + span, "done")


def install_torch(reporter: Reporter, force_cpu: bool) -> None:
    check = subprocess.run([python_exe(), "-c", "import torch; print(torch.__version__)"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, encoding="utf-8", errors="replace")
    if check.returncode == 0:
        reporter.log(f"PyTorch already installed ({check.stdout.strip()})")
        return
    if force_cpu or not has_nvidia_gpu():
        reporter.log("No NVIDIA GPU detected - installing the CPU build of PyTorch")
        index = "https://download.pytorch.org/whl/cpu"
    else:
        reporter.log("NVIDIA GPU detected - installing PyTorch with CUDA support")
        index = f"https://download.pytorch.org/whl/{CUDA_TAG}"
    pip_install([f"torch=={TORCH_VERSION}", f"torchaudio=={TORCH_VERSION}",
                 "--index-url", index], reporter, "PyTorch")


def download_models(preset: str, reporter: Reporter) -> None:
    spec = PRESET_MODELS.get(preset, PRESET_MODELS["balanced"])
    args = [python_exe(), "-m", "vtrans.download",
            "--asr-model", spec["asr"],
            "--translate-model", spec["translate"]]
    for voice in spec["voices"]:
        args += ["--piper-voice", voice]

    env = dict(os.environ)
    env["PYTHONPATH"] = str(INSTALL_ROOT / "src")
    reporter.log(f"Downloading the '{preset}' models. This is the slow part.")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            cwd=str(INSTALL_ROOT), env=env)
    assert proc.stdout is not None
    tail: List[str] = []
    for line in proc.stdout:
        tail.append(line.rstrip())
        del tail[:-30]
        if line.strip():
            reporter.log("  " + line.strip()[:110])
    if proc.wait() != 0:
        raise RuntimeError("Model download failed:\n" + "\n".join(tail[-15:]))


def verify(reporter: Reporter) -> None:
    script = (
        "import sys\n"
        "bad = []\n"
        "for mod, label in [('torch','PyTorch'), ('faster_whisper','faster-whisper'),\n"
        "                   ('transformers','transformers'), ('piper.voice','piper-tts'),\n"
        "                   ('soundfile','soundfile'), ('PySide6.QtWidgets','PySide6'),\n"
        "                   ('vtrans','vtrans'), ('vtrans_gui','vtrans_gui')]:\n"
        "    try: __import__(mod)\n"
        "    except Exception as e: bad.append(f'{label}: {e}')\n"
        "print('CUDA:', __import__('torch').cuda.is_available())\n"
        "if bad:\n"
        "    print('MISSING:'); [print(' ', b) for b in bad]; sys.exit(1)\n"
        "print('ALL OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(INSTALL_ROOT / "src")
    proc = subprocess.run([python_exe(), "-c", script], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                          cwd=str(INSTALL_ROOT), env=env)
    reporter.log(proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError("Verification failed - some components are missing.")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Video Translator setup")
    parser.add_argument("--preset", default="balanced",
                        choices=sorted(PRESET_MODELS), help="which models to fetch")
    parser.add_argument("--cpu-only", action="store_true",
                        help="install the CPU build of PyTorch even if a GPU is present")
    parser.add_argument("--skip-models", action="store_true",
                        help="install packages only; models download on first use")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable progress for the installer UI")
    args = parser.parse_args(argv)

    reporter = Reporter(as_json=args.json)
    try:
        reporter.step("Preparing", 0.02)
        pip_install(["--upgrade", "pip", "setuptools", "wheel"], reporter, "pip")

        reporter.step("ffmpeg", 0.05)
        install_ffmpeg(reporter, 0.05, 0.10)

        reporter.step("PyTorch", 0.16)
        install_torch(reporter, args.cpu_only)

        reporter.step("Dependencies", 0.45)
        pip_install(["-r", str(INSTALL_ROOT / "requirements.txt")],
                    reporter, "pipeline dependencies")

        reporter.step("Interface", 0.55)
        pip_install(["PySide6-Essentials==6.8.1"], reporter, "the user interface")

        if not args.skip_models:
            reporter.step("Models", 0.60)
            download_models(args.preset, reporter)

        reporter.step("Verifying", 0.95)
        verify(reporter)

        reporter.step("Finished", 1.0)
        reporter.done(True, "Setup complete.")
        return 0
    except Exception as exc:  # noqa: BLE001 - reported to the installer UI
        reporter.done(False, str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
