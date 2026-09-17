"""Logging, subprocess and ffmpeg helpers shared by every stage."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from . import binaries
from .progress import Cancelled
from .runtime import subprocess_flags

LOG = logging.getLogger("vtrans")


class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def setup_logging(verbose: bool = False, logfile: Path | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s %(levelname)-7s %(message)s"
    datefmt = "%H:%M:%S"
    handlers: List[logging.Handler] = []

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(fmt, datefmt))
    handlers.append(stream)

    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    # These libraries are chatty at INFO level.
    for noisy in ("faster_whisper", "transformers", "torch", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def banner(step: int, total: int, title: str) -> None:
    line = f"[{step}/{total}] {title}"
    LOG.info("%s%s%s", Colors.BOLD + Colors.BLUE, line, Colors.RESET)


@contextmanager
def timed(label: str):
    start = time.time()
    yield
    LOG.info("%s%s took %s%s", Colors.DIM, label, fmt_duration(time.time() - start), Colors.RESET)


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def run(cmd: Sequence[str], *, check: bool = True, capture: bool = True,
        desc: str | None = None) -> subprocess.CompletedProcess:
    """Run a subprocess, logging the command at DEBUG level."""
    argv = [str(c) for c in cmd]
    LOG.debug("$ %s", " ".join(argv))
    try:
        proc = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=True,
            # Without this every ffmpeg call flashes a console window when the
            # pipeline runs under the GUI on Windows. No effect on Linux.
            creationflags=subprocess_flags(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{desc or argv[0]} could not be started: {argv[0]} was not found."
        ) from exc
    if check and proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-25:]
        raise RuntimeError(
            f"{desc or cmd[0]} failed (exit {proc.returncode}):\n" + "\n".join(tail)
        )
    return proc


def run_with_progress(cmd: Sequence[str], *, total_seconds: float,
                     on_progress=None, should_cancel=None,
                     desc: str | None = None) -> None:
    """Run an ffmpeg command, reporting progress and honouring cancellation.

    `-progress pipe:1 -nostats` makes ffmpeg emit `key=value` lines on stdout,
    including `out_time_us`, which is the only reliable way to know how far an
    encode has got. The caller supplies the expected duration; ffmpeg does not
    report a percentage itself.

    Parsing stdout also gives us a place to notice cancellation and terminate
    the child, which matters because the final encode is long enough that a
    user will want to stop it.
    """
    argv = [str(c) for c in cmd]
    LOG.debug("$ %s", " ".join(argv))
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=subprocess_flags(),
    )

    stderr_tail: List[str] = []

    def drain_stderr() -> None:
        # ffmpeg writes diagnostics to stderr; keep the tail for error reports
        # and stop the pipe filling up, which would deadlock the child.
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line.rstrip())
            del stderr_tail[:-40]

    drainer = threading.Thread(target=drain_stderr, daemon=True)
    drainer.start()

    cancelled = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if should_cancel and should_cancel():
                cancelled = True
                proc.terminate()
                break
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and on_progress and total_seconds > 0:
                try:
                    seconds = int(value) / 1_000_000
                except ValueError:
                    continue
                on_progress(min(1.0, seconds / total_seconds))
            elif key == "progress" and value == "end" and on_progress:
                on_progress(1.0)
    finally:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        drainer.join(timeout=2)

    if cancelled:
        raise Cancelled("cancelled by user")
    if proc.returncode != 0:
        raise RuntimeError(
            f"{desc or argv[0]} failed (exit {proc.returncode}):\n"
            + "\n".join(stderr_tail[-25:])
        )


def require_tool(name: str, hint: str = "") -> str:
    """Resolve an external tool, preferring a bundled copy over PATH."""
    return binaries.require(name, hint)


def ffprobe_json(path: Path) -> Dict[str, Any]:
    proc = run([
        binaries.ffprobe(), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ], desc="ffprobe")
    return json.loads(proc.stdout)


def media_duration(path: Path) -> float:
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    if fmt.get("duration"):
        return float(fmt["duration"])
    for stream in info.get("streams", []):
        if stream.get("duration"):
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration of {path}")


def has_video_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0
               for s in ffprobe_json(path).get("streams", []))


def has_audio_stream(path: Path) -> bool:
    return any(s.get("codec_type") == "audio" for s in ffprobe_json(path).get("streams", []))


def ffmpeg_has(kind: str, name: str) -> bool:
    """kind: 'filters' | 'encoders' | 'decoders'."""
    proc = run([binaries.ffmpeg(), "-hide_banner", f"-{kind}"], check=False)
    return any(line.split()[1:2] == [name] for line in (proc.stdout or "").splitlines() if line.strip())


def hf_cache_dir(models_dir: Path) -> str:
    """Keep every Hugging Face download inside the project's models/ directory."""
    import os
    home = Path(models_dir) / "hf"
    hub = home / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(home)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    return str(hub)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"


def chunked(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
