"""Final assembly: video + English audio + subtitles."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from . import binaries
from .binaries import IS_WINDOWS
from .utils import ffmpeg_has, has_audio_stream, run, run_with_progress

LOG = logging.getLogger("vtrans")


def pick_video_codec(configured: str, burning: bool) -> str:
    if configured and configured != "auto":
        return configured
    if not burning:
        return "copy"
    # NVENC is dramatically faster on a laptop GPU and the quality difference at
    # these settings is not visible for talking-head content.
    return "h264_nvenc" if ffmpeg_has("encoders", "h264_nvenc") else "libx264"


# ffmpeg >= 5 renamed the NVENC presets to p1..p7; 4.x only knows the old
# names, and new builds still accept them. Map x264 preset names onto that
# common set so one --preset value works everywhere.
NVENC_PRESETS = {
    "ultrafast": "fast", "superfast": "fast", "veryfast": "fast", "faster": "fast",
    "fast": "fast", "medium": "medium",
    "slow": "slow", "slower": "slow", "veryslow": "slow", "placebo": "slow",
}


def nvenc_preset(preset: str) -> str:
    return NVENC_PRESETS.get((preset or "").lower(), "medium")


# NVENC's quantiser is coarser than x264's CRF: the same number yields a much
# bigger file. +4 lands within a few percent of libx264 at the same --crf.
# (ffmpeg 4.x also silently ignores -cq in vbr mode, hence constqp.)
NVENC_QP_OFFSET = 4


def nvenc_qp(crf: int) -> int:
    return max(0, min(51, int(crf) + NVENC_QP_OFFSET))


def _escape_filter_path(path: Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument.

    ffmpeg parses filter arguments in two passes, so the separators have to be
    escaped for both. Windows needs different treatment from POSIX: a backslash
    is a path separator there, not an escape, and the drive-letter colon would
    otherwise be read as the start of the next filter option. The accepted form
    is forward slashes with the colon escaped once:

        C:\\jobs\\subs.ass   ->   C\\:/jobs/subs.ass
        /home/u/subs.ass  ->   /home/u/subs.ass   (unchanged)
    """
    text = str(path)
    if IS_WINDOWS:
        # libass and ffmpeg both accept forward slashes on Windows, and using
        # them sidesteps backslash-as-escape ambiguity entirely.
        text = text.replace("\\", "/")
        for char in (":", "'", "[", "]", ","):
            text = text.replace(char, "\\" + char)
        return text
    for char in ("\\", ":", "'", "[", "]", ","):
        text = text.replace(char, "\\" + char)
    return text


def mux(video: Path, audio: Path, out_path: Path, *, ass_path: Optional[Path] = None,
        srt_path: Optional[Path] = None, zh_srt_path: Optional[Path] = None,
        burn: bool = True, soft: bool = True, video_codec: str = "auto",
        crf: int = 20, preset: str = "medium", audio_bitrate: str = "192k",
        keep_original_audio: bool = True, duration: Optional[float] = None,
        on_progress=None, should_cancel=None) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    burning = burn and ass_path is not None and ass_path.exists()
    codec = pick_video_codec(video_codec, burning)
    has_orig_audio = keep_original_audio and has_audio_stream(video)
    container = out_path.suffix.lower()

    cmd: List[str] = [binaries.ffmpeg(), "-hide_banner", "-loglevel", "error", "-stats", "-y",
                      "-i", str(video), "-i", str(audio)]

    # The MP4/MOV muxer force-enables the first subtitle track whatever
    # -disposition says, so embedding one next to burned-in captions makes
    # players draw the same text twice. Matroska honours the flag, so there a
    # switched-off track can be carried safely.
    if soft and burning and container in (".mp4", ".mov"):
        LOG.info("Not embedding a subtitle track: %s always auto-enables it, which "
                 "would show the captions twice over the burned-in text. The .srt "
                 "files are written alongside; output to .mkv to embed one too.",
                 container.lstrip("."))
        soft = False

    soft_inputs: List[Path] = []
    if soft and container in (".mp4", ".mkv", ".mov"):
        if srt_path and srt_path.exists():
            soft_inputs.append(srt_path)
        if zh_srt_path and zh_srt_path.exists() and container == ".mkv":
            # mp4/mov only reliably carry one mov_text track per language set;
            # keep the Chinese track for Matroska where it is well supported.
            soft_inputs.append(zh_srt_path)
        for path in soft_inputs:
            cmd += ["-i", str(path)]

    if burning:
        cmd += ["-vf", f"ass='{_escape_filter_path(ass_path)}'"]

    cmd += ["-map", "0:v:0", "-map", "1:a:0"]
    if has_orig_audio:
        cmd += ["-map", "0:a:0"]
    for i in range(len(soft_inputs)):
        cmd += ["-map", str(2 + i)]

    if codec == "copy":
        cmd += ["-c:v", "copy"]
    elif codec == "h264_nvenc":
        cmd += ["-c:v", "h264_nvenc", "-preset", nvenc_preset(preset),
                "-rc", "constqp", "-qp", str(nvenc_qp(crf)), "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", codec, "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]

    cmd += ["-c:a", "aac", "-b:a", audio_bitrate, "-ac", "2"]
    cmd += ["-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English (dubbed)"]
    if has_orig_audio:
        cmd += ["-metadata:s:a:1", "language=chi", "-metadata:s:a:1", "title=Original Chinese"]
    if soft_inputs:
        cmd += ["-c:s", "mov_text" if container in (".mp4", ".mov") else "srt"]
        for i, path in enumerate(soft_inputs):
            lang = "chi" if ".zh." in path.name else "eng"
            cmd += [f"-metadata:s:s:{i}", f"language={lang}"]

    # Dispositions decide what a player switches on by itself. Every mapped
    # stream must be set explicitly: ffmpeg copies the source flags otherwise,
    # which leaves two "default" audio tracks and a subtitle track that draws
    # the same text a second time over the burned-in captions.
    cmd += ["-disposition:a:0", "default"]
    if has_orig_audio:
        cmd += ["-disposition:a:1", "0"]
    for i in range(len(soft_inputs)):
        # Only auto-enable a soft track when nothing is burned into the picture.
        enable = (not burning) and i == 0
        cmd += [f"-disposition:s:{i}", "default" if enable else "0"]
    if container in (".mp4", ".mov"):
        cmd += ["-movflags", "+faststart"]
    # Not -shortest: that counts the subtitle track too, and the last caption
    # usually ends well before the picture does, which would truncate the file.
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd.append(str(out_path))

    LOG.info("Muxing final video (video codec: %s, burn-in subtitles: %s)", codec, burning)

    if on_progress or should_cancel:
        # The encode is the longest single ffmpeg call in the run, so when a
        # front end is watching, stream ffmpeg's own progress rather than
        # leaving the bar frozen for minutes.
        cmd += ["-progress", "pipe:1", "-nostats"]
        run_with_progress(cmd, total_seconds=duration or 0.0,
                          on_progress=on_progress, should_cancel=should_cancel,
                          desc="ffmpeg (mux)")
    else:
        run(cmd, capture=False, desc="ffmpeg (mux)")
    return out_path
