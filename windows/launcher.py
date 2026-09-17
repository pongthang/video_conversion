"""What the Start-menu shortcut runs.

Two jobs beyond starting the GUI. First, put the install directory's src/ on
sys.path, because the app is installed as a source tree rather than pip-installed
into the runtime; that keeps "Repair" able to replace files without touching the
interpreter. Second, catch anything that goes wrong early enough that the GUI
cannot report it itself, and show a message box rather than dying silently:
launched from a shortcut with pythonw.exe there is no console to print to.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

INSTALL_ROOT = Path(__file__).resolve().parent.parent


def _fatal(title: str, message: str) -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    except Exception:  # noqa: BLE001 - not on Windows, or no user32
        print(f"{title}: {message}", file=sys.stderr)


def main() -> int:
    sys.path.insert(0, str(INSTALL_ROOT / "src"))
    try:
        from vtrans_gui.app import main as gui_main
    except ImportError as exc:
        _fatal(
            "Miko Video Translator",
            "The application's components are not installed correctly.\n\n"
            f"{exc}\n\n"
            "Run 'Repair Miko Video Translator' from the Start menu to restore them.",
        )
        return 1

    try:
        return gui_main()
    except Exception:  # noqa: BLE001 - last resort, there is no console
        _fatal("Miko Video Translator - unexpected error",
               "The application stopped unexpectedly.\n\n" + traceback.format_exc()[-1500:])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
