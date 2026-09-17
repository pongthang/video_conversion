"""GPU detection and VRAM-aware precision selection.

The pipeline is designed to run on a small consumer GPU (4 GB class), so the
stages run strictly one after another and each one frees its model before the
next starts.
"""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

LOG = logging.getLogger("vtrans")


@dataclass
class DeviceInfo:
    device: str           # "cuda" or "cpu"
    index: int            # cuda device index
    name: str
    total_vram_gb: float

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


def detect(preference: str = "auto") -> DeviceInfo:
    if preference == "cpu":
        return DeviceInfo("cpu", 0, "cpu", 0.0)

    try:
        import torch
    except ImportError:
        if preference == "cuda":
            raise RuntimeError("device: cuda requested but PyTorch is not installed")
        return DeviceInfo("cpu", 0, "cpu", 0.0)

    if not torch.cuda.is_available():
        if preference == "cuda":
            raise RuntimeError(
                "device: cuda requested but torch.cuda.is_available() is False. "
                "Check the NVIDIA driver and that the CUDA build of torch is installed."
            )
        LOG.info("No usable CUDA device found; running on CPU (this will be slow).")
        return DeviceInfo("cpu", 0, "cpu", 0.0)

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    return DeviceInfo("cuda", idx, props.name, props.total_memory / (1024 ** 3))


def whisper_compute_type(dev: DeviceInfo, configured: str = "auto") -> str:
    """Pick a CTranslate2 compute type that fits the available VRAM."""
    if configured and configured != "auto":
        return configured
    if not dev.is_cuda:
        return "int8"
    # large-v3 is ~3.1 GB in float16 which does not leave room for activations
    # on a 4 GB card, so anything under ~6 GB gets the quantised weights.
    return "float16" if dev.total_vram_gb >= 6.0 else "int8_float16"


def torch_dtype(dev: DeviceInfo):
    import torch
    return torch.float16 if dev.is_cuda else torch.float32


def free_vram_gb(dev: DeviceInfo) -> float:
    """VRAM currently free on the device, in GB (0.0 on CPU)."""
    if not dev.is_cuda:
        return 0.0
    try:
        import torch
        free_bytes, _total = torch.cuda.mem_get_info(dev.index)
        return free_bytes / (1024 ** 3)
    except Exception:  # pragma: no cover - driver quirks
        return dev.total_vram_gb


def free_memory() -> None:
    """Release model memory between stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:  # pragma: no cover - torch optional at this point
        pass


def describe(dev: DeviceInfo) -> str:
    if dev.is_cuda:
        return f"{dev.name} ({dev.total_vram_gb:.1f} GB VRAM)"
    return "CPU"
