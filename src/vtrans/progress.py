"""Progress reporting, cancellation and stage weighting.

The CLI is happy with log lines, but a GUI needs a number that moves. The
pipeline therefore talks to a Reporter instead of only to the logger. The
default implementation does nothing, so the CLI path is unchanged.

Overall progress is a weighted sum of the stages. The weights are not fixed:
ASR dominates a CPU run but is a fraction of a CUDA run, and muxing scales with
the video length rather than the speech content, so estimate() adjusts them
from the device and the source duration. A bar that is roughly right the whole
way through beats one that is exactly right only in hindsight.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

LOG = logging.getLogger("vtrans")


class Cancelled(Exception):
    """Raised inside a stage when the user asks to stop."""


class CancelToken:
    """Thread-safe cancellation flag.

    The GUI sets it from the UI thread; the worker checks it between items.
    Stages are only interruptible at item boundaries, so a long ffmpeg call
    is additionally killed by the process group it was started in.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled("cancelled by user")

    def reset(self) -> None:
        self._event.clear()


# Relative cost of each stage on a CUDA machine with default settings, measured
# on a 45 s clip and a 3 h video. "separate" and "fetch" are conditional and get
# zeroed by estimate() when they do not apply.
BASE_WEIGHTS: Dict[str, float] = {
    "fetch": 2.0,
    "separate": 18.0,
    "asr": 30.0,
    "segment": 0.5,
    "translate": 10.0,
    "tts": 14.0,
    "subtitles": 0.5,
    "mux": 12.0,
}


def estimate(stages: List[str], *, on_cuda: bool, separating: bool,
             is_url: bool, duration: float = 0.0) -> Dict[str, float]:
    """Normalised stage weights summing to 1.0."""
    weights = dict(BASE_WEIGHTS)

    if not separating:
        weights["separate"] = 0.0
    if not is_url:
        weights["fetch"] = 0.5          # local file: just audio extraction

    if not on_cuda:
        # The model stages fall off a cliff on CPU; ffmpeg does not.
        weights["asr"] *= 6.0
        weights["translate"] *= 5.0
        weights["separate"] *= 8.0
        weights["tts"] *= 1.5

    # Muxing is proportional to video length, while ASR and TTS follow the
    # amount of speech. On very long sources the encode is a bigger share.
    if duration > 3600:
        weights["mux"] *= 1.6

    active = {name: max(0.0, weights.get(name, 1.0)) for name in stages}
    total = sum(active.values()) or 1.0
    return {name: value / total for name, value in active.items()}


@dataclass
class Reporter:
    """Receives progress from the pipeline. Subclass or pass callbacks.

    All methods are called from the worker thread, never the UI thread.
    """

    on_overall: Optional[Callable[[float], None]] = None
    on_stage: Optional[Callable[[str, str, float], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None
    token: Optional[CancelToken] = None

    weights: Dict[str, float] = field(default_factory=dict)
    _completed: float = 0.0
    _current: str = ""
    _current_weight: float = 0.0
    _current_title: str = ""

    # -- lifecycle -------------------------------------------------------

    def configure(self, weights: Dict[str, float]) -> None:
        self.weights = weights
        self._completed = 0.0

    def stage_started(self, name: str, title: str) -> None:
        self._current = name
        self._current_title = title
        self._current_weight = self.weights.get(name, 0.0)
        self._emit_stage(0.0)
        self._emit_overall()

    def stage_progress(self, fraction: float) -> None:
        """Fraction of the *current* stage, 0.0 to 1.0."""
        fraction = min(1.0, max(0.0, fraction))
        self._emit_stage(fraction)
        self._emit_overall(fraction)

    def stage_finished(self, name: str) -> None:
        self._completed += self.weights.get(name, 0.0)
        self._current = ""
        self._current_weight = 0.0
        self._emit_overall()

    # -- messages --------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        if self.on_log:
            self.on_log(message, level)

    def warn(self, message: str) -> None:
        self.log(message, "warning")

    # -- cancellation ----------------------------------------------------

    def check_cancelled(self) -> None:
        if self.token:
            self.token.check()

    @property
    def cancelled(self) -> bool:
        return bool(self.token and self.token.cancelled)

    # -- internals -------------------------------------------------------

    def _emit_overall(self, fraction: float = 0.0) -> None:
        if not self.on_overall:
            return
        value = self._completed + self._current_weight * fraction
        self.on_overall(min(1.0, max(0.0, value)))

    def _emit_stage(self, fraction: float) -> None:
        if self.on_stage and self._current:
            self.on_stage(self._current, self._current_title, fraction)


class NullReporter(Reporter):
    """What the CLI uses: every call is a no-op, no branching at call sites."""

    def stage_started(self, name: str, title: str) -> None:
        pass

    def stage_progress(self, fraction: float) -> None:
        pass

    def stage_finished(self, name: str) -> None:
        pass

    def log(self, message: str, level: str = "info") -> None:
        pass

    def check_cancelled(self) -> None:
        pass
