"""Input acquisition, audio extraction and silence-aware chunking."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from . import binaries
from .utils import media_duration, run

LOG = logging.getLogger("vtrans")

URL_RE = re.compile(r"^(https?|ftp)://", re.IGNORECASE)


def is_url(value: str) -> bool:
    return bool(URL_RE.match(value.strip()))


def fetch_source(source: str, dest_dir: Path) -> Path:
    """Return a local media file for `source` (URL or existing path)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not is_url(source):
        path = Path(source).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {source}")
        return path

    existing = sorted(dest_dir.glob("source.*"))
    existing = [p for p in existing if p.suffix not in (".part", ".ytdl")]
    if existing:
        LOG.info("Reusing previously downloaded source: %s", existing[0].name)
        return existing[0]

    LOG.info("Downloading %s", source)
    out_tpl = str(dest_dir / "source.%(ext)s")
    # Prefer H.264 + AAC at <=1080p. YouTube's newer AV1/Opus formats cannot be
    # copied into MP4 by older ffmpeg builds, and AV1 has no hardware decode on
    # the GPUs this pipeline targets, so it would slow every later stage down.
    # The alternatives after each "/" are progressively looser fallbacks.
    fmt = ("bv*[height<=1080][vcodec^=avc1]+ba[acodec^=mp4a]/"
           "b[height<=1080][vcodec^=avc1]/"
           "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b")
    run([
        binaries.require("yt-dlp", "It is installed by setup.sh / the Windows installer."),
        "-f", fmt,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "10",
        "--newline",
        "-o", out_tpl,
        source,
    ], capture=False, desc="yt-dlp")

    downloaded = sorted(p for p in dest_dir.glob("source.*") if p.suffix not in (".part", ".ytdl"))
    if not downloaded:
        raise RuntimeError("yt-dlp finished but no output file was produced")
    return downloaded[0]


def extract_audio(src: Path, dest: Path, sample_rate: int = 16000, channels: int = 1) -> Path:
    """Extract a normalised mono PCM WAV suitable for ASR / source separation."""
    if dest.exists() and dest.stat().st_size > 0:
        LOG.info("Reusing extracted audio: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    run([
        binaries.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vn", "-sn", "-dn",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(dest),
    ], desc="ffmpeg (extract audio)")
    return dest


def detect_silences(audio: Path, noise_db: float = -35.0, min_duration: float = 0.35
                    ) -> List[Tuple[float, float]]:
    """Return [(start, end)] of silent regions using ffmpeg's silencedetect."""
    proc = run([
        binaries.ffmpeg(), "-hide_banner", "-nostats", "-i", str(audio),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ], desc="ffmpeg (silencedetect)")
    text = (proc.stderr or "") + (proc.stdout or "")

    silences: List[Tuple[float, float]] = []
    pending_start: float | None = None
    for match in re.finditer(r"silence_(start|end):\s*(-?[\d.]+)", text):
        kind, value = match.group(1), float(match.group(2))
        if kind == "start":
            pending_start = value
        elif pending_start is not None:
            silences.append((pending_start, value))
            pending_start = None
    return silences


@dataclass
class Chunk:
    index: int
    start: float
    end: float
    path: Path

    @property
    def duration(self) -> float:
        return self.end - self.start


def plan_chunks(audio: Path, chunk_minutes: float,
                search_window: float | None = None) -> List[Tuple[float, float]]:
    """Plan chunk boundaries, snapping each cut to the nearest silent stretch.

    Cutting on silence matters: a naive fixed-length cut lands mid-word and the
    ASR then mis-transcribes the first and last word of every chunk.
    """
    total = media_duration(audio)
    if chunk_minutes <= 0 or total <= chunk_minutes * 60:
        return [(0.0, total)]

    if chunk_minutes < 5.0:
        LOG.warning(
            "chunk_minutes=%.1f is very small. Whisper transcribes short chunks "
            "noticeably worse (it hallucinates at chunk starts); 20 is the tested "
            "default and 5 a sensible floor.", chunk_minutes)

    target = chunk_minutes * 60.0
    # Every tolerance scales with the chunk length, so a small --chunk-minutes
    # behaves sensibly instead of being overruled by fixed second counts.
    if search_window is None:
        search_window = min(90.0, max(2.0, target * 0.1))
    min_chunk = min(60.0, target * 0.5)
    min_tail = min(30.0, target * 0.25)
    silences = detect_silences(audio)
    LOG.debug("Found %d silent regions for chunk planning", len(silences))

    bounds: List[float] = [0.0]
    while bounds[-1] + target < total:
        ideal = bounds[-1] + target
        best: float | None = None
        best_dist = search_window
        for s_start, s_end in silences:
            mid = (s_start + s_end) / 2.0
            # Must move the boundary forward past the previous cut.
            if mid <= bounds[-1] + min_chunk:
                continue
            dist = abs(mid - ideal)
            if dist < best_dist:
                best, best_dist = mid, dist
        cut = best if best is not None else ideal
        if total - cut < min_tail:  # don't leave a sliver at the end
            break
        bounds.append(cut)
    bounds.append(total)

    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def split_audio(audio: Path, spans: List[Tuple[float, float]], dest_dir: Path,
                sample_rate: int = 16000) -> List[Chunk]:
    """Cut `audio` into WAV chunks for the given [(start, end)] spans."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    chunks: List[Chunk] = []
    for i, (start, end) in enumerate(spans):
        out = dest_dir / f"chunk_{i:03d}.wav"
        if not out.exists() or out.stat().st_size == 0:
            run([
                binaries.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                "-i", str(audio),
                "-ac", "1", "-ar", str(sample_rate), "-acodec", "pcm_s16le",
                str(out),
            ], desc="ffmpeg (split audio)")
        chunks.append(Chunk(index=i, start=start, end=end, path=out))
    return chunks
