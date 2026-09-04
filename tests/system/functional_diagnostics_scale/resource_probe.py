# © Artur Czarnecki. All rights reserved.

"""Process RSS probe without optional third-party dependencies."""

from __future__ import annotations

import os
import sys


def process_rss_bytes() -> int | None:
    if sys.platform == "win32":
        return _windows_working_set_bytes()
    return _posix_rss_bytes()


def cpu_core_count() -> int | None:
    count = os.cpu_count()
    if count is None or count < 1:
        return None
    return count


def _posix_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = usage.ru_maxrss
    if sys.platform == "darwin":
        return int(rss)
    return int(rss) * 1024


def _windows_working_set_bytes() -> int | None:
    try:
        import ctypes
        import ctypes.wintypes
    except ImportError:
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    process = ctypes.windll.kernel32.GetCurrentProcess()
    if ctypes.windll.psapi.GetProcessMemoryInfo(
        process,
        ctypes.byref(counters),
        counters.cb,
    ) == 0:
        return None
    return int(counters.WorkingSetSize)


__all__ = ["cpu_core_count", "process_rss_bytes"]
