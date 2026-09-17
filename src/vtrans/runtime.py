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


def _expose_cuda_libraries() -> None:
    dirs = _nvidia_lib_dirs()
    if not dirs:
        return
    if IS_WINDOWS:
        # Since 3.8 Windows ignores PATH for extension-module DLL loading.
        for directory in dirs:
            try:
                os.add_dll_directory(str(directory))
            except (OSError, AttributeError) as exc:
                LOG.debug("add_dll_directory(%s) failed: %s", directory, exc)
        # CTranslate2 resolves some libraries through PATH regardless, so set
        # both and let whichever mechanism applies find them.
        os.environ["PATH"] = os.pathsep.join(
            [str(d) for d in dirs] + [os.environ.get("PATH", "")]
        )
    else:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [str(d) for d in dirs] + ([existing] if existing else [])
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
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
