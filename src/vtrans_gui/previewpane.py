"""Live caption preview.

Rendering goes through ffmpeg and libass (see vtrans.preview) so what is shown
is what will be burned in. That costs a process launch, so the render is
debounced and runs on a worker thread: dragging the size slider must not block
the UI or spawn a render per pixel.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from vtrans.preview import render_placeholder, render_preview

# Long enough that dragging a slider does not queue renders, short enough that
# it still feels like a live preview.
DEBOUNCE_MS = 220


class _Signals(QObject):
    """Owned by the pane, never by the task.

    QThreadPool deletes a QRunnable as soon as run() returns. A signals object
    living on the task would therefore be destroyed while the queued
    cross-thread emission was still waiting to be delivered to the UI thread,
    which is a use-after-free and crashes the process intermittently. Giving
    the pane a single long-lived emitter removes the lifetime question
    entirely, and the serial says which render a result belongs to.
    """

    done = Signal(str, int)      # png path, serial
    failed = Signal(str, int)    # message, serial


class _RenderTask(QRunnable):
    def __init__(self, video: Optional[Path], out_png: Path, options: dict,
                 signals: _Signals, serial: int) -> None:
        super().__init__()
        self.video = video
        self.out_png = out_png
        self.options = options
        self._signals = signals
        self._serial = serial

    @Slot()
    def run(self) -> None:
        try:
            if self.video and self.video.is_file():
                render_preview(self.video, self.out_png, **self.options)
            else:
                render_placeholder(self.out_png, **self.options)
            self._signals.done.emit(str(self.out_png), self._serial)
        except Exception as exc:  # noqa: BLE001 - a preview is never fatal
            self._signals.failed.emit(str(exc), self._serial)


class PreviewPane(QWidget):
    """Shows a frame of the chosen video with sample captions burned in."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._video: Optional[Path] = None
        self._options: dict = {}
        self._serial = 0
        self._tmpdir = Path(tempfile.mkdtemp(prefix="vtrans-gui-preview-"))
        self._pool = QThreadPool(self)
        # One render at a time; a queued burst would just waste CPU.
        self._pool.setMaxThreadCount(1)

        self._signals = _Signals(self)
        self._signals.done.connect(self._on_done)
        self._signals.failed.connect(self._on_failed)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(DEBOUNCE_MS)
        self._timer.timeout.connect(self._render_now)

        self.image = QLabel("Choose a video to preview the captions")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumHeight(200)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image.setStyleSheet(
            "QLabel { background: #1e1e1e; color: #9a9a9a; border: 1px solid #3a3a3a;"
            " border-radius: 4px; }"
        )

        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setStyleSheet("QLabel { color: #8a8a8a; font-size: 11px; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.caption)

        self._pixmap: Optional[QPixmap] = None

    # ------------------------------------------------------------- public

    def set_video(self, video: Optional[Path]) -> None:
        self._video = video
        self.request(**self._options) if self._options else None

    def request(self, **options) -> None:
        """Schedule a re-render. Repeated calls inside the debounce coalesce."""
        self._options = options
        self._timer.start()

    # ------------------------------------------------------------ internals

    def _render_now(self) -> None:
        self._serial += 1
        serial = self._serial
        out_png = self._tmpdir / f"preview-{serial}.png"
        task = _RenderTask(self._video, out_png, dict(self._options),
                           self._signals, serial)
        self.caption.setText("rendering preview...")
        self._pool.start(task)

    def _on_done(self, path: str, serial: int) -> None:
        # A slower earlier render must not overwrite a newer one.
        if serial != self._serial:
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.caption.setText("preview unavailable")
            return
        self._pixmap = pixmap
        self._apply_pixmap()
        size = self._options.get("font_size", "?")
        source = "your video" if self._video else "sample background"
        self.caption.setText(f"Caption size {size} on {source} - this is exactly how it will be burned in")
        try:
            Path(path).unlink()
        except OSError:
            pass

    def _on_failed(self, message: str, serial: int) -> None:
        if serial != self._serial:
            return
        self.caption.setText(f"preview unavailable ({message.splitlines()[0][:70]})")

    def _apply_pixmap(self) -> None:
        if not self._pixmap:
            return
        scaled = self._pixmap.scaled(self.image.width(), self.image.height(),
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._apply_pixmap()

    def shutdown(self) -> None:
        """Wait for any in-flight render before the widget is torn down."""
        self._timer.stop()
        self._serial += 1          # invalidate anything still running
        self._pool.clear()
        self._pool.waitForDone(5000)
        shutil.rmtree(self._tmpdir, ignore_errors=True)
