"""Model catalogue: what each option costs and what it needs to run.

The pipeline lets you name any Whisper size or HF repo, which is right for a
command line and useless in a GUI: nothing on screen says whether large-v3 will
fit. This table carries the requirements so the UI can mark each option against
the hardware actually present.

The VRAM figures are the weights plus room for activations at the batch sizes
the pipeline uses, rounded up. They are estimates, and the UI presents them as
such; the honest ones come from device.whisper_compute_type, which quantises
automatically on small cards. Disk sizes are the download.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelOption:
    key: str                 # value written into the config
    label: str               # shown in the combo box
    vram_gb: float           # VRAM needed on the GPU path, 0 = CPU-only model
    ram_gb: float            # system RAM needed on the CPU path
    disk_gb: float           # download size
    speed: str               # relative throughput, human readable
    quality: str             # human readable
    note: str = ""

    def fits(self, vram_gb: float) -> bool:
        return vram_gb >= self.vram_gb

    def requirement_text(self, on_gpu: bool) -> str:
        if on_gpu:
            return f"{self.vram_gb:.1f} GB VRAM · {self.disk_gb:.1f} GB disk · {self.speed}"
        return f"{self.ram_gb:.1f} GB RAM · {self.disk_gb:.1f} GB disk · {self.speed} (CPU)"


# --------------------------------------------------------------------- ASR
# VRAM is for the compute type the pipeline picks: float16 above 6 GB,
# int8_float16 below, which roughly halves the weights.
ASR_MODELS: List[ModelOption] = [
    ModelOption("tiny", "Tiny - fastest, roughest", 0.8, 1.5, 0.08,
                "~12x realtime", "Low",
                "Usable for a quick check that the pipeline runs end to end."),
    ModelOption("base", "Base", 1.0, 2.0, 0.15, "~8x realtime", "Low"),
    ModelOption("small", "Small - good balance on 4 GB cards", 1.4, 3.0, 0.5,
                "~5x realtime", "Fair",
                "The practical default when large-v3 does not fit."),
    ModelOption("medium", "Medium", 2.4, 5.0, 1.5, "~2.5x realtime", "Good"),
    ModelOption("distil-large-v3", "Distil large-v3 - near-large, much faster", 2.6, 6.0, 1.5,
                "~4x realtime", "Very good",
                "Distilled from large-v3. Best quality per GB on a small card."),
    ModelOption("large-v2", "Large-v2", 3.4, 8.0, 3.1, "~1.2x realtime", "Excellent"),
    ModelOption("large-v3", "Large-v3 - best accuracy", 3.6, 8.0, 3.1,
                "~1x realtime", "Excellent",
                "Quantised to int8_float16 automatically on cards below 6 GB."),
]

# -------------------------------------------------------------- Translation
TRANSLATE_MODELS: List[ModelOption] = [
    ModelOption("opus", "Opus-MT zh-en - light and quick", 0.6, 1.5, 0.3,
                "fast", "Fair",
                "Marian model, a fraction of the size. Flatter phrasing than NLLB."),
    ModelOption("nllb", "NLLB-200 - best quality", 1.8, 4.0, 2.5,
                "moderate", "Very good",
                "The default. Handles idiom and long sentences noticeably better. "
                "Picks the 1.3B model automatically when there is VRAM free for it "
                "and it has been downloaded, otherwise the 600M."),
]

# --------------------------------------------------------------------- TTS
# Piper is ONNX on CPU by design: it is fast there and leaves the GPU free.
TTS_MODELS: List[ModelOption] = [
    ModelOption("piper", "Piper - fast, fully offline", 0.0, 1.0, 0.07,
                "~20x realtime", "Good",
                "Runs on CPU even when the rest of the run is on the GPU."),
    ModelOption("kokoro", "Kokoro-82M - most natural", 1.2, 3.0, 0.35,
                "~3x realtime", "Very good",
                "Better prosody. Needs espeak-ng for unusual words."),
]

# ------------------------------------------------------------------ Voices
# The GUI offers male/female; these are the voices behind each choice.
@dataclass(frozen=True)
class Voice:
    key: str
    label: str
    gender: str              # "male" | "female"
    backend: str             # "piper" | "kokoro"
    disk_mb: float
    note: str = ""


VOICES: List[Voice] = [
    # -- Piper ---------------------------------------------------------
    Voice("en_US-amy-medium", "Amy - US English", "female", "piper", 63,
          "Clear and even. The default female voice."),
    Voice("en_US-lessac-medium", "Lessac - US English", "female", "piper", 63,
          "Warmer, a little slower."),
    Voice("en_US-libritts_r-medium", "LibriTTS-R - US English", "female", "piper", 79,
          "Most natural of the Piper female voices, slightly less consistent."),
    Voice("en_US-ryan-high", "Ryan - US English", "male", "piper", 119,
          "Clear and neutral. The default male voice."),
    Voice("en_GB-alan-medium", "Alan - British English", "male", "piper", 63,
          "British accent, measured delivery."),
    Voice("en_GB-northern_english_male-medium", "Northern English", "male", "piper", 63,
          "Northern British accent."),
    # -- Kokoro --------------------------------------------------------
    Voice("af_heart", "Heart - US English", "female", "kokoro", 0,
          "Kokoro's most expressive female voice."),
    Voice("af_bella", "Bella - US English", "female", "kokoro", 0),
    Voice("am_michael", "Michael - US English", "male", "kokoro", 0),
    Voice("am_adam", "Adam - US English", "male", "kokoro", 0),
]

DEFAULT_VOICE = {
    ("female", "piper"): "en_US-amy-medium",
    ("male", "piper"): "en_US-ryan-high",
    ("female", "kokoro"): "af_heart",
    ("male", "kokoro"): "am_michael",
}


def voices_for(gender: str, backend: str) -> List[Voice]:
    return [v for v in VOICES if v.gender == gender and v.backend == backend]


def default_voice(gender: str, backend: str) -> str:
    return DEFAULT_VOICE.get((gender, backend), "en_US-amy-medium")


def find_option(options: List[ModelOption], key: str) -> Optional[ModelOption]:
    for option in options:
        if option.key == key:
            return option
    return None


# ------------------------------------------------------------------ Presets
@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    asr: str
    translate: str
    tts: str
    description: str


PRESETS: List[Preset] = [
    Preset("fast", "Fast", "small", "opus", "piper",
           "Quickest usable result. Fits comfortably on a 4 GB card."),
    Preset("balanced", "Balanced", "distil-large-v3", "nllb", "piper",
           "Near-best transcription at a fraction of the cost. Recommended."),
    Preset("best", "Best quality", "large-v3", "nllb", "kokoro",
           "Highest accuracy and the most natural voice. Slowest."),
]


def preset_requirements(preset: Preset) -> Dict[str, float]:
    """Peak VRAM and total download for a preset.

    Peak, not sum: the pipeline loads one model at a time and frees it before
    the next, which is what makes a 4 GB card viable at all.
    """
    asr = find_option(ASR_MODELS, preset.asr)
    mt = find_option(TRANSLATE_MODELS, preset.translate)
    tts = find_option(TTS_MODELS, preset.tts)
    parts = [p for p in (asr, mt, tts) if p]
    return {
        "peak_vram_gb": max((p.vram_gb for p in parts), default=0.0),
        "peak_ram_gb": max((p.ram_gb for p in parts), default=0.0),
        "disk_gb": sum(p.disk_gb for p in parts),
    }
