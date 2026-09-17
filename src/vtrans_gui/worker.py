"""Drives the conversion child process and turns its output into Qt signals.

QProcess rather than subprocess + QThread: it integrates with the event loop,
so there is no polling and no risk of touching widgets off the UI thread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, Signal


def _source_environment() -> QProcessEnvironment:
    """Environment shared by every child process this module starts."""
    env = QProcessEnvironment.systemEnvironment()
    # The child must see the project on sys.path when running from a source
    # checkout, where vtrans is not installed into the interpreter.
    src = str(Path(__file__).resolve().parents[1])
    existing = env.value("PYTHONPATH", "")
    separator = ";" if sys.platform == "win32" else ":"
    env.insert("PYTHONPATH", src + (separator + existing if existing else ""))
    env.insert("PYTHONUNBUFFERED", "1")
    # Force UTF-8: the child prints Chinese source text, and the Windows console
    # default (cp1252) would raise UnicodeEncodeError mid-run.
    env.insert("PYTHONIOENCODING", "utf-8")
    return env


class ConversionWorker(QObject):
    """One conversion. Construct, connect, call start()."""

    overall = Signal(float)                 # 0.0 - 1.0
    stage = Signal(str, str, float)         # name, title, fraction
    logged = Signal(str, str)               # message, level
    device = Signal(str, str, float)        # "cuda"|"cpu", name, VRAM GB
    finished = Signal(dict)                 # the terminal protocol message
    failed = Signal(str)                    # human readable reason

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._buffer = ""
        self._terminal: Optional[Dict] = None
        self._cancelling = False

    # ------------------------------------------------------------ control

    def start(self, *, source: str, out_path: Path, overrides: Dict,
              cpu_fallback: bool = True, force_from: Optional[str] = None) -> None:
        args = [
            "-m", "vtrans.jobrunner",
            "--input", source,
            "--out", str(out_path),
            "--overrides", json.dumps(overrides),
        ]
        if not cpu_fallback:
            args.append("--no-cpu-fallback")
        if force_from:
            args += ["--force-from", force_from]

        proc = QProcess(self)
        # Keep the streams apart: stdout is the protocol, stderr is the log.
        proc.setProcessChannelMode(QProcess.SeparateChannels)
        proc.setProcessEnvironment(_source_environment())
        proc.readyReadStandardOutput.connect(self._read_stdout)
        proc.readyReadStandardError.connect(self._read_stderr)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_error)

        self._proc = proc
        self._buffer = ""
        self._terminal = None
        self._cancelling = False
        proc.start(sys.executable, args)

    def cancel(self) -> None:
        """Ask the child to stop at the next safe point."""
        if not self._proc or self._proc.state() == QProcess.NotRunning:
            return
        self._cancelling = True
        self._proc.write(b"cancel\n")

    def kill(self) -> None:
        if self._proc and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    @property
    def running(self) -> bool:
        return bool(self._proc and self._proc.state() != QProcess.NotRunning)

    # ------------------------------------------------------------ plumbing

    def _read_stdout(self) -> None:
        if not self._proc:
            return
        self._buffer += bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._handle(line)

    def _handle(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            # Anything that is not protocol is a library writing to stdout.
            self.logged.emit(line, "debug")
            return

        kind = message.get("t")
        if kind == "overall":
            self.overall.emit(float(message.get("value", 0.0)))
        elif kind == "stage":
            self.stage.emit(message.get("name", ""), message.get("title", ""),
                            float(message.get("frac", 0.0)))
        elif kind == "log":
            self.logged.emit(message.get("msg", ""), message.get("level", "info"))
        elif kind == "device":
            self.device.emit(message.get("device", ""), message.get("name", ""),
                             float(message.get("vram_gb", 0.0)))
        elif kind in ("done", "error", "cancelled"):
            self._terminal = message

    def _read_stderr(self) -> None:
        if not self._proc:
            return
        text = bytes(self._proc.readAllStandardError()).decode("utf-8", "replace")
        for line in text.splitlines():
            if line.strip():
                self.logged.emit(line.rstrip(), "stderr")

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.FailedToStart:
            self.failed.emit(
                "Could not start the conversion process. The Python runtime "
                "appears to be missing or incomplete - try reinstalling."
            )

    def _on_finished(self, exit_code: int, status: QProcess.ExitStatus) -> None:
        self._read_stdout()
        terminal = self._terminal

        if terminal and terminal.get("t") == "done":
            self.finished.emit(terminal)
            return
        if terminal and terminal.get("t") == "cancelled" or self._cancelling:
            self.finished.emit({"t": "cancelled"})
            return
        if terminal and terminal.get("t") == "error":
            self.failed.emit(terminal.get("msg", "The conversion failed."))
            return

        # No terminal message: the child died without reporting. This is the
        # case the subprocess design exists for - CTranslate2 calls abort() on
        # some CUDA failures, which no exception handler inside it can catch.
        if status == QProcess.CrashExit or exit_code < 0 or exit_code in (134, 139):
            self.failed.emit(
                "The conversion process stopped unexpectedly (exit "
                f"{exit_code}). This is usually a GPU or driver problem. "
                "Switching Device to CPU in the options will avoid it."
            )
        else:
            self.failed.emit(f"The conversion process exited with code {exit_code}.")


class PreflightWorker(QObject):
    """Runs the setup steps a conversion needs, in order, before it starts.

    Two kinds of step: pip-installing an optional backend, and downloading
    models. Both are sequential subprocesses whose progress is not usefully
    quantifiable - huggingface_hub and pip each write their own progress in
    forms not worth parsing - so the UI shows an indeterminate bar and the log,
    which is honest about what is known.

    A single worker rather than one per kind, because the steps have to happen
    in order: installing Kokoro and then downloading its model is two commands,
    and the second is pointless if the first failed.
    """

    logged = Signal(str, str)
    step = Signal(str)               # human-readable description of the step
    finished = Signal(bool, str)     # ok, message

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._proc: Optional[QProcess] = None
        self._steps: List[Tuple[str, List[str]]] = []
        self._index = 0
        self._tail: List[str] = []
        self._cancelled = False

    def start(self, steps: List[Tuple[str, List[str]]]) -> None:
        """steps: [(description, argv-after-the-interpreter), ...]"""
        self._steps = list(steps)
        self._index = 0
        self._cancelled = False
        self._run_next()

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc and self._proc.state() != QProcess.NotRunning:
            self._proc.kill()

    @property
    def running(self) -> bool:
        return bool(self._proc and self._proc.state() != QProcess.NotRunning)

    # ------------------------------------------------------------ internals

    def _run_next(self) -> None:
        if self._cancelled:
            return
        if self._index >= len(self._steps):
            self.finished.emit(True, "")
            return

        description, args = self._steps[self._index]
        self.step.emit(description)
        self.logged.emit(description, "info")

        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.setProcessEnvironment(_source_environment())
        proc.readyReadStandardOutput.connect(self._read)
        proc.finished.connect(self._on_step_finished)
        proc.errorOccurred.connect(
            lambda _e: self.finished.emit(False, "Could not start the setup step."))
        self._proc = proc
        self._tail = []
        proc.start(sys.executable, args)

    def _read(self) -> None:
        if not self._proc:
            return
        text = bytes(self._proc.readAllStandardOutput()).decode("utf-8", "replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            self._tail.append(line.rstrip())
            del self._tail[:-40]
            # pip is extremely chatty; only the lines that show movement are
            # worth putting in front of someone waiting.
            if line.startswith(("Collecting", "Downloading", "Installing",
                                "Successfully", "  ", "Models directory",
                                "Whisper", "Hugging Face")):
                self.logged.emit(line.rstrip()[:130], "info")

    def _on_step_finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read()
        if self._cancelled:
            return
        if exit_code != 0:
            description = self._steps[self._index][0]
            self.finished.emit(
                False,
                f"{description} failed.\n\n" + "\n".join(self._tail[-10:]))
            return
        self._index += 1
        self._run_next()
