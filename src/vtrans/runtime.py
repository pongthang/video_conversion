"""Process environment setup shared by the CLI and the GUI.

convert_video.sh used to do all of this in bash before exec'ing Python: point
the model caches inside the project, size the thread pools, and put the CUDA
libraries that ship as pip wheels on the loader path. The GUI has no shell in
front of it, and Windows has no LD_LIBRARY_PATH at all, so the logic lives here
and both front ends call prepare() exactly once at start-up.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

from .binaries import IS_WINDOWS, project_root

LOG = logging.getLogger("vtrans")

_PREPARED = False

# Wheels that carry the CUDA shared libraries faster-whisper/CTranslate2 dlopen
# at runtime. They are installed next to torch but never added to the loader
# path by pip, so without this CTranslate2 fails with a bare "cannot open
# shared object file" the moment a CUDA model is loaded.
_CUDA_LIB_MODULES = (
    "nvidia.cublas.lib",
    "nvidia.cudnn.lib",
    "nvidia.cuda_runtime.lib",
    "nvidia.cuda_nvrtc.lib",
    "nvidia.cufft.lib",
)


def _nvidia_lib_dirs() -> List[Path]:
    """Directories inside the installed nvidia-* wheels that hold the libraries.

    On Windows the DLLs sit in a 'bin' subdirectory rather than 'lib'; the
    module still imports as nvidia.<pkg>.lib, so probe both.
    """
    import importlib

    dirs: List[Path] = []
    for module_name in _CUDA_LIB_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - the package simply is not installed
            continue
        base = Path(module.__file__).parent if module.__file__ else None
        if base is None:
            continue
        for candidate in (base, base.parent / "bin"):
            if candidate.is_dir() and candidate not in dirs:
                dirs.append(candidate)
    return dirs


def _preload_shared_libraries(dirs: List[Path]) -> int:
    """dlopen the CUDA libraries into the global namespace, by absolute path.

    Setting LD_LIBRARY_PATH from inside a running process does nothing: glibc
    reads it once, at exec time, so a value assigned to os.environ here is only
    inherited by children. convert_video.sh got away with exporting it because
    it did so before starting Python at all, but the GUI has no wrapper script.

    Loading each library explicitly with RTLD_GLOBAL sidesteps the search path
    entirely. By the time CTranslate2 dlopens "libcudnn_ops.so.9" it is already
    resident and its symbols are visible, so the request resolves regardless of
    where the file lives.

    Interdependencies mean order matters and is not worth hardcoding, so this
    retries the failures until a pass makes no further progress.
    """
    import ctypes

    pending: List[Path] = []
    for directory in dirs:
        pending.extend(sorted(directory.glob("*.so*")))
    if not pending:
        return 0

    loaded = 0
    while pending:
        failed: List[Path] = []
        for lib in pending:
            try:
                ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                failed.append(lib)
        if len(failed) == len(pending):
            # No progress this pass: the rest genuinely cannot load (wrong
            # driver, missing dependency). CTranslate2 will fall back or fail
            # with its own message, which is more informative than ours.
            LOG.debug("Could not preload: %s", [f.name for f in failed])
            break
        pending = failed
    return loaded


def _expose_cuda_libraries() -> None:
    dirs = _nvidia_lib_dirs()
    if not dirs:
        return

    if IS_WINDOWS:
        # add_dll_directory is the supported runtime mechanism on Windows and
        # does take effect immediately, unlike LD_LIBRARY_PATH on Linux.
        for directory in dirs:
            try:
                os.add_dll_directory(str(directory))
            except (OSError, AttributeError) as exc:
                LOG.debug("add_dll_directory(%s) failed: %s", directory, exc)
        os.environ["PATH"] = os.pathsep.join(
            [str(d) for d in dirs] + [os.environ.get("PATH", "")]
        )
    else:
        # Still exported so that child processes (demucs, yt-dlp) inherit it.
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(d) for d in dirs] + ([existing] if existing else [])
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
        # ...but this process needs them resident, not merely findable.
        count = _preload_shared_libraries(dirs)
        LOG.debug("Preloaded %d CUDA shared libraries", count)

    LOG.debug("CUDA library directories: %s", [str(d) for d in dirs])


def _default_thread_count() -> int:
    """Leave a couple of cores for ffmpeg; oversubscribing slows the run down."""
    cpus = os.cpu_count() or 2
    return max(1, cpus - 2) if cpus > 2 else 1


def prepare(models_dir: Path | None = None) -> None:
    """Set up caches, thread limits and the CUDA loader path. Idempotent."""
    global _PREPARED
    if _PREPARED:
        return
    _PREPARED = True

    root = project_root()
    models = Path(models_dir) if models_dir else root / "models"

    # Keep every download inside the project / install directory so that
    # uninstalling really removes everything and nothing lands in %USERPROFILE%.
    os.environ.setdefault("HF_HOME", str(models / "hf"))
    os.environ.setdefault("TORCH_HOME", str(models / "torch"))
    os.environ.setdefault("OMP_NUM_THREADS", str(_default_thread_count()))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Progress bars from huggingface_hub corrupt a GUI log pane and add nothing
    # to a file log; the pipeline reports its own progress.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # Add the local bin/ directory (populated by the Windows installer) to PATH
    # so that child processes such as yt-dlp's ffmpeg lookup also find it.
    bin_dir = root / "bin"
    if bin_dir.is_dir():
        os.environ["PATH"] = os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")])

    _expose_cuda_libraries()


def subprocess_flags() -> int:
    """Creation flags that stop a console window flashing for every ffmpeg call.

    Harmless on Linux, where the attribute does not exist and 0 is passed.
    """
    if IS_WINDOWS:
        return getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0)
    return 0


def describe_environment() -> str:
    lines = [
        f"python      {sys.version.split()[0]} ({sys.executable})",
        f"platform    {sys.platform}",
        f"project     {project_root()}",
        f"HF_HOME     {os.environ.get('HF_HOME', '-')}",
        f"threads     {os.environ.get('OMP_NUM_THREADS', '-')}",
    ]
    return "\n".join(lines)
