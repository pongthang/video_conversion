"""Run one conversion as a child process, reporting progress as JSON lines.

The GUI does not run the pipeline in a thread. CTranslate2 responds to some GPU
failures by calling abort(): a missing cuDNN takes the whole process down
without raising anything Python can catch. In a thread that kills the GUI and
loses the log with it; in a child process the parent sees a non-zero exit code,
keeps the log it already collected, and can offer to retry on CPU.

Protocol, one JSON object per line on stdout:

    {"t": "overall", "value": 0.42}
    {"t": "stage",   "name": "asr", "title": "Transcribing...", "frac": 0.5}
    {"t": "log",     "level": "info", "msg": "..."}
    {"t": "done",    "output": "/path/to/output.mp4", "demoted": ["tts"]}
    {"t": "error",   "msg": "...", "traceback": "..."}

Cancellation arrives as the line "cancel" on stdin, rather than a signal:
Windows has no SIGINT delivery to a specific child that works the same way.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from .config import Config
from .pipeline import Pipeline, job_id
from .progress import Cancelled, CancelToken, Reporter
from .runtime import prepare
from .utils import setup_logging

_WRITE_LOCK = threading.Lock()


def emit(**payload: Any) -> None:
    """Write one protocol line. Flushed immediately so the GUI stays current."""
    with _WRITE_LOCK:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        sys.stdout.flush()


# The pipeline colours its stage banners for the terminal. A GUI text pane
# renders those bytes literally, as "[1m[34m[4/8] ...", so strip them on the
# way out. The CLI and the run.log are untouched and keep their colour.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class EmitHandler(logging.Handler):
    """Forwards the pipeline's log records to the parent as protocol lines."""

    def emit(self, record: logging.LogRecord) -> None:  # noqa: A003
        try:
            emit(t="log", level=record.levelname.lower(),
                 msg=_ANSI.sub("", record.getMessage()))
        except Exception:  # noqa: BLE001 - logging must never break the run
            pass


def _watch_stdin(token: CancelToken) -> None:
    """Set the cancel flag when the parent asks. Runs on a daemon thread."""
    try:
        for line in sys.stdin:
            if line.strip().lower() == "cancel":
                token.cancel()
                emit(t="log", level="warning", msg="Cancelling after the current step...")
                return
    except Exception:  # noqa: BLE001 - parent closed the pipe
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GUI job runner (internal)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    parser.add_argument("--overrides", help="JSON object of dotted config keys")
    parser.add_argument("--no-cpu-fallback", action="store_true")
    parser.add_argument("--force-from")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config)
    if args.overrides:
        cfg.apply_overrides(json.loads(args.overrides))
    prepare(cfg.resolve_dir("general.models_dir"))

    out_path = Path(args.out).expanduser().resolve()
    work_dir = cfg.resolve_dir("general.work_dir") / job_id(args.input)
    setup_logging(False, logfile=work_dir / "run.log")
    # Route log records to the parent as protocol messages, and drop the
    # console handler setup_logging installed. Leaving it in place would send
    # every line to stderr as well, which the parent also captures and shows -
    # the GUI log then lists everything twice.
    for handler in list(logging.getLogger().handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler):
            logging.getLogger().removeHandler(handler)
    logging.getLogger("vtrans").addHandler(EmitHandler())

    token = CancelToken()
    threading.Thread(target=_watch_stdin, args=(token,), daemon=True).start()

    reporter = Reporter(
        on_overall=lambda value: emit(t="overall", value=round(value, 4)),
        on_stage=lambda name, title, frac: emit(t="stage", name=name, title=title,
                                                frac=round(frac, 4)),
        on_log=lambda msg, level: emit(t="log", level=level, msg=msg),
        token=token,
    )

    pipeline = Pipeline(
        cfg, args.input, out_path,
        force_from=args.force_from,
        reporter=reporter,
        token=token,
        cpu_fallback=not args.no_cpu_fallback,
    )

    # Tell the parent what hardware this run is actually on, so the window can
    # show it rather than leaving the user to infer it from the log.
    emit(t="device", device=pipeline.dev.device, name=pipeline.dev.name,
         vram_gb=round(pipeline.dev.total_vram_gb, 1))

    try:
        final = pipeline.run()
    except Cancelled:
        emit(t="cancelled")
        return 130
    except KeyboardInterrupt:
        emit(t="cancelled")
        return 130
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        emit(t="error", msg=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc())
        return 1

    emit(t="done", output=str(final), demoted=pipeline.fallback.demoted,
         work_dir=str(pipeline.job_dir), subtitles=str(pipeline.en_srt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
