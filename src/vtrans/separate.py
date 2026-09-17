"""Optional vocal/instrumental separation with Demucs.

Replacing the whole soundtrack throws away the music and effects, which is the
single most noticeable quality loss in a naive dub. Separating first lets the
original bed stay under the English voice.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from .device import DeviceInfo
from .utils import run

LOG = logging.getLogger("vtrans")


def separate_background(audio: Path, out_dir: Path, models_dir: Path, dev: DeviceInfo,
                        model: str = "htdemucs", segment: int = 7) -> Optional[Path]:
    """Return the path to the instrumental (no-vocals) stem, or None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "background.wav"
    if final.exists() and final.stat().st_size > 0:
        LOG.info("Reusing separated background: %s", final.name)
        return final

    if shutil.which("demucs") is None:
        LOG.warning("demucs is not installed; skipping background separation. "
                    "Install it with: pip install demucs")
        return None

    env_home = models_dir / "torch"
    env_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(env_home))

    stage = out_dir / "demucs"
    cmd = [
        "demucs", "--two-stems", "vocals", "-n", model,
        "-o", str(stage), "--filename", "{stem}.{ext}",
        "-d", "cuda" if dev.is_cuda else "cpu",
    ]
    # Demucs keeps a whole segment in memory; 7 s fits comfortably in 4 GB.
    if segment:
        cmd += ["--segment", str(segment)]
    cmd.append(str(audio))

    LOG.info("Separating vocals from background with Demucs (%s) - this takes a while", model)
    try:
        run(cmd, capture=False, desc="demucs")
    except RuntimeError as exc:
        LOG.warning("Demucs failed (%s); continuing without a background bed.", exc)
        return None

    candidates = list(stage.rglob("no_vocals.*"))
    if not candidates:
        LOG.warning("Demucs produced no 'no_vocals' stem; continuing without background.")
        return None

    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(candidates[0]), "-ac", "2", "-acodec", "pcm_s16le", str(final)],
        desc="ffmpeg (background convert)")
    shutil.rmtree(stage, ignore_errors=True)
    LOG.info("Background bed ready: %s", final.name)
    return final
