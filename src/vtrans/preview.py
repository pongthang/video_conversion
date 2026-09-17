"""Render a still frame with sample captions burned in.

Choosing a caption size from a number is guesswork: 22 is comfortable on a
talking-head 1080p clip and far too small on a busy 4K screen recording. So the
GUI shows the real thing instead, on a frame from the user's own video.

The preview is faithful rather than approximate because it goes through the
same code as the final render: subtitles.write_ass produces the style block,
and ffmpeg's ass filter draws it with libass. A Qt-drawn mock-up would use
different font metrics, different outline geometry and different line breaking,
which is exactly the detail someone is trying to judge here.

Scale is preserved too. The ASS header pins PlayResY to 288 and libass scales
that to the frame height, so a font size is a fraction of picture height (18 is
about 6%) no matter the resolution. Rendering the preview at 720 px wide
therefore looks proportionally identical to the 4K burn.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from . import binaries
from .subtitles import Cue, wrap_text, write_ass
from .utils import media_duration, run

LOG = logging.getLogger("vtrans")

# Sample caption shown when there is no real transcript yet. Its length is
# chosen to match a realistic cue: build_cues caps captions at max_lines lines
# of max_line_chars, so a sample much longer than that would be re-wrapped by
# libass and would misrepresent how the real captions sit on the picture.
SAMPLE_TEXT = "This is what your subtitles will look like at this size."


def _pick_frame_time(video: Path, fallback: float = 5.0) -> float:
    """A frame far enough in to be representative, but cheap to seek to."""
    try:
        duration = media_duration(video)
    except Exception:  # noqa: BLE001 - preview must never break the UI
        return fallback
    if duration <= 2.0:
        return duration / 2.0
    # A tenth of the way in usually clears titles and fade-ins without needing
    # a long seek.
    return min(max(duration * 0.1, 2.0), 60.0)


def render_preview(video: Path, out_png: Path, *, font: str = "DejaVu Sans",
                   font_size: int = 18, outline: int = 2, shadow: int = 0,
                   margin_v: int = 28, max_line_chars: int = 48, max_lines: int = 2,
                   text: Optional[Sequence[str]] = None,
                   width: int = 854, at_seconds: Optional[float] = None) -> Path:
    """Burn sample captions onto one frame of `video` and write it as a PNG.

    Returns the PNG path. Raises RuntimeError if ffmpeg fails, which the caller
    should treat as "no preview available" rather than as a fatal error.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sample = " ".join(text) if text else SAMPLE_TEXT
    seek = _pick_frame_time(video) if at_seconds is None else at_seconds

    # One cue covering the frame's timestamp, wrapped exactly as the real
    # subtitle stage would wrap it.
    wrapped = wrap_text(sample, max_line_chars, max_lines)
    cues: List[Cue] = [Cue(start=0.0, end=10.0, text=wrapped)]

    with tempfile.TemporaryDirectory(prefix="vtrans-preview-") as tmp:
        ass_path = Path(tmp) / "preview.ass"
        write_ass(cues, ass_path, font=font, size=font_size, outline=outline,
                  shadow=shadow, margin_v=margin_v)

        from .mux import _escape_filter_path
        # Scale first, then draw: libass renders against the scaled frame, which
        # is what keeps the preview proportional to the final burn.
        vf = (f"scale={width}:-2,"
              f"ass='{_escape_filter_path(ass_path)}'")
        run([
            binaries.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{seek:.3f}", "-i", str(video),
            "-vf", vf, "-frames:v", "1", "-update", "1",
            str(out_png),
        ], desc="ffmpeg (subtitle preview)")

    if not out_png.exists() or out_png.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no preview frame")
    return out_png


def render_placeholder(out_png: Path, *, font: str = "DejaVu Sans", font_size: int = 18,
                       outline: int = 2, shadow: int = 0, margin_v: int = 28,
                       max_line_chars: int = 48, max_lines: int = 2,
                       text: Optional[Sequence[str]] = None,
                       width: int = 854, height: int = 480) -> Path:
    """Same preview against a neutral grey card, for when no video is loaded yet."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sample = " ".join(text) if text else SAMPLE_TEXT
    wrapped = wrap_text(sample, max_line_chars, max_lines)

    with tempfile.TemporaryDirectory(prefix="vtrans-preview-") as tmp:
        ass_path = Path(tmp) / "preview.ass"
        write_ass([Cue(start=0.0, end=10.0, text=wrapped)], ass_path, font=font,
                  size=font_size, outline=outline, shadow=shadow, margin_v=margin_v)
        from .mux import _escape_filter_path
        run([
            binaries.ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=0x3a3a3a:s={width}x{height}",
            "-vf", f"ass='{_escape_filter_path(ass_path)}'",
            "-frames:v", "1", "-update", "1", str(out_png),
        ], desc="ffmpeg (subtitle preview)")
    return out_png
