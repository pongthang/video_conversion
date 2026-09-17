"""Per-stage GPU failure recovery.

A 4 GB card runs this pipeline comfortably only because the stages load their
models one at a time. That still leaves plenty of ways for CUDA to fail midway:
another process grabs the VRAM, the driver mismatches the cuDNN the wheels
shipped, or a long sentence pushes a batch over the edge.

The response is per stage, not per run. If Demucs cannot fit, Whisper usually
still can, so demoting the whole job to CPU would cost far more time than the
one stage that actually failed. Each stage is therefore wrapped here: on a
recognised GPU failure its model is rebuilt on CPU and the stage runs once more.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, TypeVar

from .device import DeviceInfo, free_memory
from .progress import Cancelled, Reporter

LOG = logging.getLogger("vtrans")

T = TypeVar("T")

# Substrings that identify an error as "the GPU could not do this", as opposed
# to a bug or a bad input, which must not be retried. Matched case-insensitively
# against the exception text.
_GPU_ERROR_MARKERS = (
    "out of memory",
    "cuda",
    "cudnn",
    "cublas",
    "cufft",
    "nvrtc",
    "no kernel image is available",
    "device-side assert",
    "cuda error",
    "libcudart",
    "gpu",
    "nvidia",
)

# Markers that mean "this would have failed on CPU too": never retry these.
_NOT_GPU_MARKERS = (
    "no such file or directory",
    "permission denied",
    "connection",
    "timed out",
    "404",
)


def looks_like_gpu_failure(exc: BaseException) -> bool:
    """Whether `exc` is plausibly the GPU's fault and worth retrying on CPU."""
    if isinstance(exc, (Cancelled, KeyboardInterrupt)):
        return False

    # torch names its OOM explicitly; catch it without importing torch when
    # torch is not what raised.
    if type(exc).__name__ in ("OutOfMemoryError", "CudaError", "CublasError"):
        return True

    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _NOT_GPU_MARKERS):
        return False
    return any(marker in text for marker in _GPU_ERROR_MARKERS)


def cpu_device() -> DeviceInfo:
    return DeviceInfo("cpu", 0, "cpu", 0.0)


@dataclass
class FallbackPolicy:
    """How to react when a stage fails on the GPU.

    enabled=False reproduces the old behaviour: the error propagates and the
    run stops, which is what someone debugging a GPU problem wants.
    """

    enabled: bool = True
    reporter: Optional[Reporter] = None
    # Stages that ended up on CPU, for the final report.
    demoted: List[str] = field(default_factory=list)

    def run_stage(self, name: str, dev: DeviceInfo,
                  work: Callable[[DeviceInfo], T]) -> T:
        """Run `work(dev)`, retrying once on CPU if the GPU is at fault."""
        if not dev.is_cuda:
            return work(dev)

        try:
            return work(dev)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, then filtered
            if not self.enabled or not looks_like_gpu_failure(exc):
                raise

            message = (
                f"GPU failed during '{name}' ({type(exc).__name__}: "
                f"{str(exc).strip().splitlines()[0][:160]}). Retrying on CPU - "
                f"this stage will be slower."
            )
            LOG.warning(message)
            if self.reporter:
                self.reporter.warn(message)

            free_memory()
            self.demoted.append(name)
            return work(cpu_device())

    def summary(self) -> str:
        if not self.demoted:
            return ""
        return "Ran on CPU after a GPU failure: " + ", ".join(self.demoted)
