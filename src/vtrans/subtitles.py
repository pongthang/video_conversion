"""SRT and ASS subtitle generation.

Caption timings follow the *English* audio, not the Chinese source, so the text
on screen matches what the viewer hears.
"""
from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .segment import Sentence
from .sync import Clip

LOG = logging.getLogger("vtrans")

MIN_CUE = 0.9          # seconds a caption stays up even if the line is short
MAX_CUE = 8.0
CUE_GAP = 0.04         # keeps consecutive cues from touching


@dataclass
class Cue:
    start: float
    end: float
    text: str


def wrap_text(text: str, max_chars: int = 42, max_lines: int = 2) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False,
                          break_on_hyphens=False)
    if len(lines) <= max_lines:
        return "\n".join(lines)
    # Too long for the allowed lines: widen just enough to fit them.
    widened = textwrap.wrap(text, width=max(max_chars, len(text) // max_lines + 6),
                            break_long_words=False, break_on_hyphens=False)
    return "\n".join(widened[:max_lines]) if len(widened) > max_lines else "\n".join(widened)


def build_cues(sentences: List[Sentence], clips: Optional[List[Clip]] = None, *,
               use_target: bool = True, max_line_chars: int = 42,
               max_lines: int = 2, total_duration: float | None = None) -> List[Cue]:
    clip_by_index = {c.index: c for c in (clips or [])}
    cues: List[Cue] = []

    for sentence in sentences:
        text = (sentence.target if use_target else sentence.source) or ""
        text = text.strip()
        if not text:
            continue
        clip = clip_by_index.get(sentence.index)
        start = clip.start if clip else sentence.start
        end = clip.end if clip else sentence.end
        if end - start < MIN_CUE:
            end = start + MIN_CUE
        end = min(end, start + MAX_CUE)
        cues.append(Cue(start=start, end=end, text=wrap_text(text, max_line_chars, max_lines)))

    cues.sort(key=lambda c: c.start)
    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start - CUE_GAP:
            cues[i].end = max(cues[i].start + 0.2, cues[i + 1].start - CUE_GAP)
    if total_duration and cues:
        cues[-1].end = min(cues[-1].end, total_duration)
    return cues


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(cues: List[Cue], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i, cue in enumerate(cues, start=1):
            fh.write(f"{i}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{cue.text}\n\n")
    LOG.info("Wrote %s (%d cues)", path.name, len(cues))
    return path


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {play_x}
PlayResY: {play_y}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{size},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _ass_escape(text: str) -> str:
    return (text.replace("\\", "\\\\")
                .replace("{", "\\{")
                .replace("}", "\\}")
                .replace("\n", "\\N"))


def write_ass(cues: List[Cue], path: Path, *, font: str = "DejaVu Sans", size: int = 22,
              outline: int = 2, shadow: int = 0, margin_v: int = 28,
              play_res: tuple[int, int] = (384, 288)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(ASS_HEADER.format(font=font, size=size, outline=outline, shadow=shadow,
                                   margin_v=margin_v, play_x=play_res[0], play_y=play_res[1]))
        for cue in cues:
            fh.write(f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},"
                     f"Default,,0,0,0,,{_ass_escape(cue.text)}\n")
    LOG.info("Wrote %s (%d cues)", path.name, len(cues))
    return path
