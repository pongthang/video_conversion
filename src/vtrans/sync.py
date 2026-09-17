"""Fit the synthesised English speech onto the original timeline.

This is the part that decides whether a dub feels synced or not. For every
sentence we know the slot it must occupy (its own span plus most of the silence
that follows it). English is usually longer than Chinese, so a line frequently
overruns. Two levers are used, in this order:

  1. Re-synthesise slightly faster (TTS `length_scale`) - sounds natural.
  2. Time-stretch the rendered audio with rubberband/atempo - pitch preserving.

Crucially, start times are never shifted. Shifting would make each line's error
accumulate and the dub would drift further out of sync as the video goes on.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

from .segment import Sentence, strip_for_tts
from .tts import TTSEngine
from .utils import fmt_duration, run

LOG = logging.getLogger("vtrans")


@dataclass
class Clip:
    index: int
    start: float          # placement on the timeline (== sentence start)
    end: float            # actual end after fitting
    text: str
    speed: float          # total speed factor applied (1.0 = untouched)
    overflow: float       # seconds it spills past its allowed budget
    budget: float

    def to_dict(self) -> dict:
        return asdict(self)


def _has_rubberband() -> bool:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return "rubberband" in (proc.stdout or "")


def _atempo_chain(factor: float) -> str:
    """atempo only accepts 0.5-2.0 per instance; chain for anything beyond."""
    parts = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


class TimeStretcher:
    """Pitch-preserving time stretch through ffmpeg."""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        self.use_rubberband = _has_rubberband()
        self._tmpdir = Path(tempfile.mkdtemp(prefix="vtrans-stretch-"))
        if not self.use_rubberband:
            LOG.info("rubberband filter unavailable; using atempo for time stretching")

    def stretch(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """factor > 1 makes the audio shorter (faster)."""
        if audio.size == 0 or abs(factor - 1.0) < 0.01:
            return audio
        src = self._tmpdir / "in.wav"
        dst = self._tmpdir / "out.wav"
        sf.write(src, audio, self.sample_rate, subtype="PCM_16")
        af = (f"rubberband=tempo={factor:.6f}:pitch=1:transients=crisp"
              if self.use_rubberband else _atempo_chain(factor))
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(src), "-af", af, str(dst)], desc="ffmpeg (time stretch)")
        out, _ = sf.read(dst, dtype="float32", always_2d=False)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def close(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def _apply_fade(audio: np.ndarray, sample_rate: int, fade_seconds: float) -> np.ndarray:
    n = int(fade_seconds * sample_rate)
    if n <= 0 or audio.size < 2 * n:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    audio = audio.copy()
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


def _trim_silence(audio: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    """Drop leading/trailing near-silence so placement lands on the first phoneme."""
    if audio.size == 0:
        return audio
    loud = np.nonzero(np.abs(audio) > threshold)[0]
    if loud.size == 0:
        return audio[:0]
    return audio[loud[0]:loud[-1] + 1]


def render_track(sentences: List[Sentence], engine: TTSEngine, cfg,
                 total_duration: float, out_wav: Path,
                 progress=None) -> List[Clip]:
    """Synthesise every line and place it on a silent canvas of the full length."""
    sr = engine.sample_rate
    base_ls = float(cfg.get("tts.length_scale", 1.0))
    min_ls = float(cfg.get("sync.min_length_scale", 0.78))
    max_speedup = float(cfg.get("sync.max_speedup", 1.45))
    gap_usage = float(cfg.get("sync.gap_usage", 0.9))
    allow_overlap = bool(cfg.get("sync.allow_overlap", True))
    fade = float(cfg.get("sync.fade", 0.012))

    canvas = np.zeros(int(np.ceil(total_duration * sr)) + sr, dtype=np.float32)
    stretcher = TimeStretcher(sr)
    clips: List[Clip] = []
    stats = {"natural": 0, "resynth": 0, "stretched": 0, "overflow": 0}

    try:
        for i, sentence in enumerate(sentences):
            text = strip_for_tts(sentence.target or "")
            if not text:
                continue

            next_start = sentences[i + 1].start if i + 1 < len(sentences) else total_duration
            gap = max(0.0, next_start - sentence.end)
            budget = max(0.35, sentence.duration + gap * gap_usage)

            audio = _trim_silence(engine.synth(text, length_scale=base_ls))
            if audio.size == 0:
                continue
            duration = audio.size / sr
            speed = 1.0

            if duration > budget:
                needed = min(duration / budget, max_speedup)
                # Do as much as possible in the synthesiser: it re-renders the
                # prosody instead of squeezing an existing waveform.
                synth_factor = min(needed, base_ls / min_ls)
                if synth_factor > 1.01:
                    audio = _trim_silence(engine.synth(text, length_scale=base_ls / synth_factor))
                    stats["resynth"] += 1
                    duration = audio.size / sr
                    speed *= synth_factor
                remaining = duration / budget
                if remaining > 1.01:
                    remaining = min(remaining, max_speedup / max(speed, 1e-6))
                    if remaining > 1.01:
                        audio = stretcher.stretch(audio, remaining)
                        stats["stretched"] += 1
                        duration = audio.size / sr
                        speed *= remaining
            else:
                stats["natural"] += 1

            overflow = max(0.0, duration - budget)
            if overflow > 0.05:
                stats["overflow"] += 1
                if not allow_overlap:
                    keep = int(budget * sr)
                    audio = audio[:keep]
                    duration = audio.size / sr
                    overflow = 0.0

            audio = _apply_fade(audio, sr, fade)
            offset = int(max(0.0, sentence.start) * sr)
            end_idx = offset + audio.size
            if end_idx > canvas.size:
                canvas = np.pad(canvas, (0, end_idx - canvas.size))
            canvas[offset:end_idx] += audio

            clips.append(Clip(
                index=sentence.index,
                start=sentence.start,
                end=sentence.start + duration,
                text=text,
                speed=round(speed, 4),
                overflow=round(overflow, 3),
                budget=round(budget, 3),
            ))

            if progress and (i % 10 == 0 or i == len(sentences) - 1):
                progress(i + 1, len(sentences))
    finally:
        stretcher.close()

    peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
    if peak > 0.99:
        canvas *= 0.99 / peak
        LOG.debug("Dub canvas peak-limited from %.2f", peak)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, canvas, sr, subtype="PCM_16")

    LOG.info("Rendered %d clips: %d at natural pace, %d re-synthesised faster, "
             "%d time-stretched, %d still overrun",
             len(clips), stats["natural"], stats["resynth"],
             stats["stretched"], stats["overflow"])
    if stats["overflow"] > len(clips) * 0.25 and len(clips):
        LOG.warning(
            "%.0f%% of lines exceed their slot. Consider --max-speedup 1.6 or a "
            "shorter translation style.", 100.0 * stats["overflow"] / len(clips))
    LOG.info("Dub track written to %s (%s)", out_wav.name, fmt_duration(canvas.size / sr))
    return clips


def mix_with_background(dub_wav: Path, background_wav: Optional[Path], out_wav: Path,
                        background_gain_db: float = -3.0, loudness_lufs: float = -16.0,
                        sample_rate: int = 48000, duration: Optional[float] = None) -> Path:
    """Mix the dub over the separated instrumental (if any) and normalise loudness."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # The canvas is deliberately padded past the end so no clip is clipped during
    # assembly; trim it back to the picture length here.
    trim = ["-t", f"{duration:.3f}"] if duration else []
    if background_wav and background_wav.exists():
        LOG.info("Mixing dub with background bed at %.1f dB", background_gain_db)
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(dub_wav), "-i", str(background_wav),
            "-filter_complex",
            # amix gained its `normalize` option in ffmpeg 4.4, so instead let it
            # halve both inputs and undo that with volume=2. The dub/background
            # ratio is unaffected either way, and loudnorm sets the final level.
            f"[1:a]volume={background_gain_db}dB[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0[sum];"
            f"[sum]volume=2.0[mix];"
            f"[mix]loudnorm=I={loudness_lufs}:TP=-1.5:LRA=11,aresample={sample_rate}[out]",
            "-map", "[out]", "-ac", "2", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le", *trim, str(out_wav),
        ], desc="ffmpeg (mix)")
    else:
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(dub_wav),
            "-af", f"loudnorm=I={loudness_lufs}:TP=-1.5:LRA=11,aresample={sample_rate}",
            "-ac", "2", "-ar", str(sample_rate),
            "-acodec", "pcm_s16le", *trim, str(out_wav),
        ], desc="ffmpeg (normalise)")
    return out_wav
