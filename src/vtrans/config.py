"""Configuration loading and merging.

Precedence (lowest to highest): config/default.yaml < --config file < CLI flags.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config:
    """Dict wrapper with dotted-path access: cfg.get("asr.model")."""

    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def load(cls, extra_path: str | os.PathLike | None = None) -> "Config":
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if extra_path:
            with open(extra_path, "r", encoding="utf-8") as fh:
                data = _deep_merge(data, yaml.safe_load(fh) or {})
        return cls(data)

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply {dotted.path: value} pairs, skipping None (= flag not given)."""
        for path, value in overrides.items():
            if value is not None:
                self.set(path, value)

    def resolve_dir(self, path: str) -> Path:
        """Resolve a config path relative to the project root."""
        p = Path(self.get(path))
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def dump(self) -> str:
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)
