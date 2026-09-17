"""Serialise GPU work across pipeline runs.

Stages inside one run already load exactly one model at a time and free it
before the next starts, so a single run has the whole card to itself. Two runs
started at once would halve that, and on a 4 GB card the second one usually
dies with an out-of-memory error partway through a long job.

An advisory lock makes the runs queue instead: the second waits for the first
to finish rather than failing an hour in.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("vtrans")

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore

try:  # Windows
    import msvcrt
except ImportError:
    msvcrt = None  # type: ignore


class GpuLock:
    """Exclusive, advisory, released automatically if the holder dies."""

    def __init__(self, path: Path, enabled: bool = True, poll_seconds: float = 5.0):
        self.path = path
        self.enabled = enabled
        self.poll_seconds = poll_seconds
        self._fh = None

    def _try_acquire(self) -> bool:
        if fcntl is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except OSError:
                return False
        if msvcrt is not None:  # pragma: no cover - Windows
            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                return False
        return True  # no locking primitive: proceed rather than block forever

    def __enter__(self) -> "GpuLock":
        if not self.enabled:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+")
        if not self._try_acquire():
            LOG.warning("Another conversion is using the GPU. Waiting for it to "
                        "finish so this run gets the whole card "
                        "(Ctrl-C to abort, or pass --no-gpu-lock to run anyway).")
            waited = 0.0
            while not self._try_acquire():
                time.sleep(self.poll_seconds)
                waited += self.poll_seconds
                if waited % 300 < self.poll_seconds:
                    LOG.info("  still waiting for the GPU (%.0f min)", waited / 60)
            LOG.info("GPU is free; continuing.")
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(f"{os.getpid()}\n")
        self._fh.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        finally:
            self._fh.close()
            self._fh = None


def maybe_lock(work_root: Path, use_cuda: bool, enabled: bool = True) -> GpuLock:
    """Only meaningful when the run will actually touch the GPU."""
    return GpuLock(work_root / ".gpu.lock", enabled=enabled and use_cuda)
