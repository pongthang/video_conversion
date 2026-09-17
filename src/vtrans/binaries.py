"""Locating the external tools the pipeline shells out to.

On Linux they come from the system (apt install ffmpeg) and from the virtualenv
that setup.sh builds. On Windows there is no package manager to lean on, so the
installer drops static builds into <project>/bin and yt-dlp lands in the
standalone runtime's Scripts directory. Everything that used to hardcode the
bare name "ffmpeg" goes through here instead, so both layouts work unchanged.

Resolution order for every tool:
    1. an explicit override env var (VTRANS_FFMPEG, VTRANS_FFPROBE, VTRANS_YTDLP)
    2. <project root>/bin/           - what the Windows installer populates
    3. the directory of the running interpreter, and its Scripts/bin sibling
    4. PATH
"""
from __future__ import annotations

import functools
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

IS_WINDOWS = os.name == "nt"

# Tool name -> environment variable that overrides its location outright.
_OVERRIDES = {
    "ffmpeg": "VTRANS_FFMPEG",
    "ffprobe": "VTRANS_FFPROBE",
    "yt-dlp": "VTRANS_YTDLP",
}


def project_root() -> Path:
    """The directory holding config/, models/ and (on Windows) bin/.

    src/vtrans/binaries.py -> up three levels. When the app is frozen into an
    exe there is no src/ layer, so fall back to the executable's directory.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _candidate_dirs() -> List[Path]:
    dirs = [project_root() / "bin"]
    exe_dir = Path(sys.executable).resolve().parent
    dirs.append(exe_dir)
    # A standalone CPython puts console scripts in Scripts/ (Windows) or bin/.
    dirs.append(exe_dir / ("Scripts" if IS_WINDOWS else "bin"))
    dirs.append(exe_dir.parent / ("Scripts" if IS_WINDOWS else "bin"))
    return dirs


def _executable_names(name: str) -> List[str]:
    if not IS_WINDOWS:
        return [name]
    # shutil.which already consults PATHEXT, but a direct file probe does not.
    return [name + ext for ext in (".exe", ".cmd", ".bat", "")]


@functools.lru_cache(maxsize=None)
def find(name: str) -> Optional[str]:
    """Return the full path to `name`, or None if it cannot be found."""
    override = _OVERRIDES.get(name)
    if override:
        value = os.environ.get(override)
        if value and Path(value).is_file():
            return str(Path(value).resolve())

    for directory in _candidate_dirs():
        for candidate in _executable_names(name):
            path = directory / candidate
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())

    found = shutil.which(name)
    return str(Path(found).resolve()) if found else None


def require(name: str, hint: str = "") -> str:
    """Return the full path to `name`, raising a helpful error if it is absent."""
    path = find(name)
    if path:
        return path
    raise RuntimeError(
        f"Required tool '{name}' was not found.\n"
        f"  Looked in: {', '.join(str(d) for d in _candidate_dirs())}, and PATH.\n"
        f"  {hint}".rstrip()
    )


_FFMPEG_HINT = (
    "Linux: sudo apt install ffmpeg. "
    "Windows: re-run the installer, which downloads a static build into bin/."
)


def ffmpeg() -> str:
    return require("ffmpeg", _FFMPEG_HINT)


def ffprobe() -> str:
    return require("ffprobe", _FFMPEG_HINT)


def ytdlp() -> Optional[str]:
    """yt-dlp is only needed for URL inputs, so absence is not fatal here."""
    return find("yt-dlp")


def reset_cache() -> None:
    """Forget resolved paths (used by tests and after the installer adds bin/)."""
    find.cache_clear()
