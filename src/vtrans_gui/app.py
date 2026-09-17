"""Application entry point."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from vtrans import binaries
from vtrans.runtime import prepare

from .settings import APP_NAME
from .theme import STYLESHEET


def _check_prerequisites(app: QApplication) -> bool:
    """ffmpeg is the one dependency the app cannot do anything without."""
    try:
        binaries.ffmpeg()
        binaries.ffprobe()
    except RuntimeError as exc:
        QMessageBox.critical(
            None, "Setup incomplete",
            f"{exc}\n\nRe-run the installer to restore the bundled ffmpeg."
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    prepare()

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName("Video Translator")
    app.setStyleSheet(STYLESHEET)

    icon_path = Path(__file__).resolve().parent / "assets" / "app.ico"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))

    if not _check_prerequisites(app):
        return 1

    # Imported after prepare() so the window's hardware probe sees the
    # environment the pipeline will actually run in.
    from .main_window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
