"""Persisted UI choices.

Kept next to the user's data rather than in the install directory, which on
Windows is usually read-only for a standard account.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

LOG = logging.getLogger("vtrans")

APP_NAME = "MikoVideoTranslator"


def config_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_output_dir() -> Path:
    videos = Path.home() / "Videos"
    return videos if videos.is_dir() else Path.home()


@dataclass
class Settings:
    preset: str = "balanced"
    asr_model: str = "distil-large-v3"
    translate_backend: str = "nllb"
    tts_backend: str = "piper"
    voice_gender: str = "female"
    voice: str = "en_US-amy-medium"
    device: str = "auto"            # auto | cuda | cpu
    cpu_fallback: bool = True
    keep_background: bool = False
    keep_original_audio: bool = True
    subtitles_mode: str = "both"
    font_size: int = 18
    margin_v: int = 28
    max_line_chars: int = 48
    output_dir: str = ""
    last_input_dir: str = ""
    advanced_open: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        path = config_dir() / "settings.json"
        if not path.is_file():
            return cls(output_dir=str(default_output_dir()))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            LOG.warning("Could not read settings (%s); using defaults", exc)
            return cls(output_dir=str(default_output_dir()))
        known = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in known}
        settings = cls(**clean)
        if not settings.output_dir:
            settings.output_dir = str(default_output_dir())
        return settings

    def save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            LOG.warning("Could not save settings: %s", exc)
