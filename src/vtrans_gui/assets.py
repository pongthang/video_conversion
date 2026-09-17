"""Which model files are actually on disk, and what it takes to get the rest.

The installer fetches the models for the tier the user picked, but the app lets
them choose others afterwards: a different voice, a larger Whisper. Without this
check that choice fails in the middle of a run, at the stage that needs the
file, after the user has already waited through transcription. So the window
asks here first and downloads anything missing up front.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from vtrans.download import FASTER_WHISPER_REPOS
from vtrans.translate import BACKEND_DEFAULT_MODELS

# A real weight file, as opposed to the config and tokenizer files that arrive
# first and make a half-finished download look complete.
MIN_WEIGHT_BYTES = 1_000_000


@dataclass(frozen=True)
class MissingAsset:
    kind: str            # "voice" | "whisper" | "translate"
    key: str             # what to pass to vtrans.download
    label: str           # shown to the user
    size_mb: float

    @property
    def download_args(self) -> List[str]:
        return {
            "voice": ["--piper-voice", self.key],
            "whisper": ["--asr-model", self.key],
            "translate": ["--translate-model", self.key],
        }[self.kind]


def piper_voice_present(voice: str, models_dir: Path) -> bool:
    onnx = models_dir / "piper" / f"{voice}.onnx"
    config = models_dir / "piper" / f"{voice}.onnx.json"
    return (onnx.is_file() and onnx.stat().st_size > MIN_WEIGHT_BYTES
            and config.is_file())


def _hf_repo_present(repo_id: str, cache_root: Path) -> bool:
    """True when a Hugging Face repo has a fully downloaded weight file."""
    folder = cache_root / ("models--" + repo_id.replace("/", "--"))
    if not folder.is_dir():
        return False
    blobs = folder / "blobs"
    # An interrupted download leaves a .incomplete file next to the blob.
    if blobs.is_dir() and any(blobs.glob("*.incomplete")):
        return False
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    for revision in snapshots.iterdir():
        if not revision.is_dir():
            continue
        for entry in revision.iterdir():
            if not entry.name.endswith((".bin", ".safetensors", ".onnx")):
                continue
            try:
                if entry.resolve().stat().st_size > MIN_WEIGHT_BYTES:
                    return True
            except OSError:
                continue
    return False


def whisper_present(model: str, models_dir: Path) -> bool:
    repo = FASTER_WHISPER_REPOS.get(model, model)
    return _hf_repo_present(repo, models_dir / "whisper")


def translate_present(model: str, models_dir: Path) -> bool:
    return _hf_repo_present(model, models_dir / "hf" / "hub")


def resolve_translate_model(backend: str) -> str:
    """The concrete repo a backend needs before a run can start.

    The nllb backend's configured model is "auto": at run time it picks the
    largest tier that fits free VRAM and is already downloaded. What has to be
    present for the run to work at all is therefore the smallest tier, the one
    it falls back to. Requiring the larger one here would make the app download
    5.5 GB that the machine may not even have the VRAM to use.
    """
    model = BACKEND_DEFAULT_MODELS.get(backend, "")
    if model == "auto":
        from vtrans.translate import NLLB_TIERS
        return NLLB_TIERS[-1][0]
    return model


def missing_for(*, voice: str, tts_backend: str, asr_model: str,
                translate_backend: str, models_dir: Path,
                voice_size_mb: float = 65.0,
                asr_size_mb: float = 0.0) -> List[MissingAsset]:
    """Everything the chosen settings need that is not on disk yet."""
    missing: List[MissingAsset] = []

    # Kokoro voices all live in one repo, so the voice name is not separately
    # downloadable; the backend check below covers it.
    if tts_backend == "piper" and not piper_voice_present(voice, models_dir):
        missing.append(MissingAsset("voice", voice, f"voice '{voice}'", voice_size_mb))

    if not whisper_present(asr_model, models_dir):
        missing.append(MissingAsset("whisper", asr_model,
                                    f"transcription model '{asr_model}'", asr_size_mb))

    if translate_backend != "none":
        model = resolve_translate_model(translate_backend)
        if model and not translate_present(model, models_dir):
            missing.append(MissingAsset("translate", model,
                                        f"translation model '{model}'", 0.0))
    return missing


def summarise(missing: List[MissingAsset]) -> str:
    if not missing:
        return ""
    known = sum(item.size_mb for item in missing)
    names = ", ".join(item.label for item in missing)
    if known > 0:
        return f"{names} (about {known:.0f} MB)"
    return names
