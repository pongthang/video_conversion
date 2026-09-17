"""What this machine can actually run.

Detection has to work before the heavy dependencies are imported, because the
GUI draws its hardware panel at start-up and importing torch costs seconds. So
nvidia-smi is tried first and torch only as a fallback.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from vtrans.runtime import subprocess_flags


@dataclass
class Hardware:
    has_gpu: bool
    gpu_name: str
    vram_gb: float
    cpu_count: int
    ram_gb: float
    driver: str = ""

    @property
    def short_gpu_name(self) -> str:
        """"NVIDIA GeForce GTX 1650" -> "GTX 1650": the vendor prefix is
        the same on every card and only pushes the useful part out of view."""
        name = self.gpu_name
        for prefix in ("NVIDIA GeForce ", "NVIDIA ", "GeForce "):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name

    @property
    def summary(self) -> str:
        if self.has_gpu:
            return f"{self.short_gpu_name} · {self.vram_gb:.1f} GB VRAM"
        return "No CUDA GPU detected"

    @property
    def cpu_summary(self) -> str:
        return f"{self.cpu_count} cores · {self.ram_gb:.0f} GB RAM"


def _system_ram_gb() -> float:
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / (1024 ** 3)
    except (ValueError, OSError):
        pass
    try:  # Windows
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return 0.0


def _probe_nvidia_smi() -> Optional[tuple]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        proc = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=10, creationflags=subprocess_flags(),
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return None
    name = parts[0]
    try:
        # nvidia-smi reports MiB.
        vram = float(re.sub(r"[^0-9.]", "", parts[1])) / 1024.0
    except ValueError:
        vram = 0.0
    driver = parts[2] if len(parts) > 2 else ""
    return name, vram, driver


def detect() -> Hardware:
    cpu_count = os.cpu_count() or 1
    ram = _system_ram_gb()

    probe = _probe_nvidia_smi()
    if probe:
        name, vram, driver = probe
        return Hardware(True, name, vram, cpu_count, ram, driver)

    # nvidia-smi missing does not prove there is no usable GPU: ask torch, but
    # only if it is already importable without a long delay.
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return Hardware(True, props.name, props.total_memory / (1024 ** 3),
                            cpu_count, ram)
    except Exception:  # noqa: BLE001
        pass

    return Hardware(False, "", 0.0, cpu_count, ram)
