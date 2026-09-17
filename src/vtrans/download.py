"""Pre-download every model into ./models so conversion runs fully offline.

Called by setup.sh; also usable directly:
    python -m vtrans.download --asr-model large-v3 --piper-voice en_US-lessac-medium
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from .config import Config
from .tts import PIPER_VOICES, piper_voice_urls
from .utils import human_size

FASTER_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def _prefer_classic_cdn() -> None:
    """Hugging Face's Xet transfer path measured 12x slower here than the plain
    CDN (0.26 MB/s vs 3.2 MB/s on the same 5.5 GB file), which turns a half-hour
    download into most of a day. Export HF_HUB_DISABLE_XET=0 to opt back in.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def _print(msg: str) -> None:
    print(f"  {msg}", flush=True)


def download_hf_repo(repo_id: str, models_dir: Path, allow_patterns=None) -> None:
    from huggingface_hub import snapshot_download

    from .utils import hf_cache_dir
    cache = hf_cache_dir(models_dir)
    _print(f"Hugging Face: {repo_id}")
    snapshot_download(repo_id=repo_id, cache_dir=cache, allow_patterns=allow_patterns,
                      max_workers=4)


def download_whisper(model: str, models_dir: Path) -> None:
    repo = FASTER_WHISPER_REPOS.get(model)
    if repo is None:
        _print(f"'{model}' is not a known Whisper size; assuming it is a HF repo id")
        repo = model
    cache = models_dir / "whisper"
    cache.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    _print(f"Whisper: {repo}")
    snapshot_download(repo_id=repo, cache_dir=str(cache), max_workers=4,
                      allow_patterns=["*.bin", "*.json", "*.txt", "*.model"])


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        _print(f"{dest.name} already present ({human_size(dest.stat().st_size)})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    _print(f"downloading {dest.name}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    _print(f"  -> {human_size(dest.stat().st_size)}")


def download_piper_voice(voice: str, models_dir: Path) -> None:
    onnx_url, json_url = piper_voice_urls(voice)
    _fetch(onnx_url, models_dir / "piper" / f"{voice}.onnx")
    _fetch(json_url, models_dir / "piper" / f"{voice}.onnx.json")


def download_demucs(model: str, models_dir: Path) -> None:
    import os
    os.environ.setdefault("TORCH_HOME", str(models_dir / "torch"))
    (models_dir / "torch").mkdir(parents=True, exist_ok=True)
    try:
        from demucs.pretrained import get_model
        _print(f"Demucs: {model}")
        get_model(model)
    except Exception as exc:  # noqa: BLE001
        _print(f"skipped Demucs ({exc})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-download pipeline models")
    parser.add_argument("--config", help="extra YAML config")
    parser.add_argument("--asr-model", help="whisper model to fetch")
    parser.add_argument("--translate-model", help="translation model repo to fetch")
    parser.add_argument("--piper-voice", action="append", help="piper voice (repeatable)")
    parser.add_argument("--kokoro", action="store_true", help="also fetch the Kokoro TTS model")
    parser.add_argument("--demucs", action="store_true", help="also fetch the Demucs model")
    parser.add_argument("--models-dir", help="override models directory")
    args = parser.parse_args(argv)

    _prefer_classic_cdn()

    cfg = Config.load(args.config)
    if args.models_dir:
        cfg.set("general.models_dir", args.models_dir)
    models_dir = cfg.resolve_dir("general.models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models directory: {models_dir}")

    asr_model = args.asr_model or cfg.get("asr.model")
    download_whisper(asr_model, models_dir)

    translate_model = args.translate_model or cfg.get("translate.model")
    if translate_model in ("auto", "", None):
        # setup.sh fetches the baseline model; the larger tier is opt-in because
        # it is another ~5.5 GB.
        from .translate import NLLB_TIERS
        translate_model = NLLB_TIERS[-1][0]
    if translate_model and cfg.get("translate.backend") != "none":
        download_hf_repo(translate_model, models_dir)

    voices = args.piper_voice or [cfg.get("tts.voice")]
    for voice in voices:
        if voice not in PIPER_VOICES:
            print(f"  warning: unknown piper voice '{voice}', skipping", file=sys.stderr)
            continue
        download_piper_voice(voice, models_dir)

    if args.kokoro:
        download_hf_repo("hexgrad/Kokoro-82M", models_dir)

    if args.demucs:
        download_demucs(cfg.get("separate.model", "htdemucs"), models_dir)

    total = sum(f.stat().st_size for f in models_dir.rglob("*")
                if f.is_file() and not f.is_symlink())
    print(f"\nAll models ready ({human_size(total)} in {models_dir})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
