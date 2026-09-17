"""English speech synthesis.

Backends:
  piper  - Rhasspy Piper (ONNX, MIT, CPU-fast, fully offline)     [default]
  kokoro - Kokoro-82M (Apache-2.0, noticeably more natural prosody)

Both expose a `length_scale` knob (>1 slower, <1 faster). Synthesising directly
at the required rate sounds much better than time-stretching afterwards, and the
sync stage relies on that.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .utils import hf_cache_dir

LOG = logging.getLogger("vtrans")

PIPER_VOICE_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# name -> path fragment inside the piper-voices repo
PIPER_VOICES = {
    "en_US-lessac-medium": "en/en_US/lessac/medium/en_US-lessac-medium",
    "en_US-lessac-high": "en/en_US/lessac/high/en_US-lessac-high",
    "en_US-amy-medium": "en/en_US/amy/medium/en_US-amy-medium",
    "en_US-ryan-high": "en/en_US/ryan/high/en_US-ryan-high",
    "en_US-libritts_r-medium": "en/en_US/libritts_r/medium/en_US-libritts_r-medium",
    "en_GB-alan-medium": "en/en_GB/alan/medium/en_GB-alan-medium",
    "en_GB-northern_english_male-medium":
        "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium",
}


def piper_voice_urls(voice: str) -> Tuple[str, str]:
    if voice not in PIPER_VOICES:
        raise KeyError(
            f"Unknown Piper voice '{voice}'. Known: {', '.join(sorted(PIPER_VOICES))}. "
            "Any voice from https://huggingface.co/rhasspy/piper-voices works if you "
            "drop its .onnx and .onnx.json into models/piper/."
        )
    frag = PIPER_VOICES[voice]
    return f"{PIPER_VOICE_BASE}/{frag}.onnx", f"{PIPER_VOICE_BASE}/{frag}.onnx.json"


class TTSEngine(ABC):
    sample_rate: int = 22050

    @abstractmethod
    def synth(self, text: str, length_scale: float = 1.0) -> np.ndarray:
        """Return mono float32 audio in [-1, 1] for `text`."""

    def close(self) -> None:
        pass


class PiperEngine(TTSEngine):
    def __init__(self, voice: str, models_dir: Path, use_cuda: bool = False,
                 noise_scale: float = 0.667, noise_w: float = 0.8,
                 sentence_silence: float = 0.1):
        from piper.voice import PiperVoice

        model_path = models_dir / "piper" / f"{voice}.onnx"
        config_path = models_dir / "piper" / f"{voice}.onnx.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {model_path}. Run ./setup.sh (or "
                f"python -m vtrans.download --piper-voice {voice})."
            )

        LOG.info("Loading Piper voice %s", voice)
        try:
            self.voice = PiperVoice.load(str(model_path), config_path=str(config_path),
                                         use_cuda=use_cuda)
        except TypeError:  # older/newer signatures without use_cuda
            self.voice = PiperVoice.load(str(model_path), config_path=str(config_path))

        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence
        cfg = getattr(self.voice, "config", None)
        self.sample_rate = int(getattr(cfg, "sample_rate", 22050))
        self._api = self._detect_api()
        LOG.debug("Piper API flavour: %s, sample rate %d", self._api, self.sample_rate)

    def _detect_api(self) -> str:
        if hasattr(self.voice, "synthesize_stream_raw"):
            return "stream_raw"
        if hasattr(self.voice, "synthesize"):
            return "chunks"
        raise RuntimeError("Unsupported piper-tts version: no known synthesis method")

    def synth(self, text: str, length_scale: float = 1.0) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)

        if self._api == "stream_raw":
            raw = b"".join(self.voice.synthesize_stream_raw(
                text,
                length_scale=length_scale,
                noise_scale=self.noise_scale,
                noise_w=self.noise_w,
                sentence_silence=self.sentence_silence,
            ))
            return _pcm16_to_float(raw)

        # piper-tts >= 1.3: synthesize() yields AudioChunk objects.
        from piper import SynthesisConfig  # type: ignore
        syn = SynthesisConfig(
            length_scale=length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w,
        )
        parts = []
        for chunk in self.voice.synthesize(text, syn_config=syn):
            self.sample_rate = int(getattr(chunk, "sample_rate", self.sample_rate))
            parts.append(_pcm16_to_float(chunk.audio_int16_bytes))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        tail = np.zeros(int(self.sentence_silence * self.sample_rate), dtype=np.float32)
        return np.concatenate(parts + [tail])


class KokoroEngine(TTSEngine):
    def __init__(self, voice: str, models_dir: Path, device: str = "cpu"):
        from kokoro import KPipeline

        hf_cache_dir(models_dir)
        LOG.info("Loading Kokoro (voice=%s) on %s", voice, device)
        self.pipeline = KPipeline(lang_code=voice[0] if voice else "a", device=device)
        self.voice = voice
        self.sample_rate = 24000

    def synth(self, text: str, length_scale: float = 1.0) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        speed = float(np.clip(1.0 / max(length_scale, 1e-3), 0.5, 2.0))
        parts = []
        for result in self.pipeline(text, voice=self.voice, speed=speed):
            audio = result[2] if isinstance(result, tuple) else getattr(result, "audio", None)
            if audio is None:
                continue
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            parts.append(arr.astype(np.float32).reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _pcm16_to_float(raw: bytes) -> np.ndarray:
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def build_engine(cfg, models_dir: Path, device: str = "cpu") -> TTSEngine:
    backend = cfg.get("tts.backend", "piper")
    if backend == "piper":
        return PiperEngine(
            voice=cfg.get("tts.voice", "en_US-lessac-medium"),
            models_dir=models_dir,
            # Piper's ONNX graph is small; CPU is reliably fast and leaves the
            # GPU free. CUDA is only worth it for the "high" quality voices.
            use_cuda=False,
            noise_scale=float(cfg.get("tts.noise_scale", 0.667)),
            noise_w=float(cfg.get("tts.noise_w", 0.8)),
            sentence_silence=float(cfg.get("tts.sentence_silence", 0.1)),
        )
    if backend == "kokoro":
        return KokoroEngine(
            voice=cfg.get("tts.kokoro_voice", "af_heart"),
            models_dir=models_dir,
            device=device,
        )
    raise ValueError(f"Unknown tts.backend '{backend}' (expected piper or kokoro)")
