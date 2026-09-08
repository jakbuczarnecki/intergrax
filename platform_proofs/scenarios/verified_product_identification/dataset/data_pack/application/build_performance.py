"""Runtime performance sampling for VPI Data Pack production builds."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DataPackBuildPerformanceSnapshot:
    peak_host_ram_mb: float | None
    peak_vram_mb: float | None


class DataPackBuildPerformanceMonitor:
    """Best-effort peak memory sampling; unavailable metrics remain None."""

    def __init__(self) -> None:
        self._peak_host_ram_mb: float | None = None
        self._peak_vram_mb: float | None = None

    def sample(self) -> None:
        host_ram = _host_ram_mb()
        if host_ram is not None:
            self._peak_host_ram_mb = max(self._peak_host_ram_mb or 0.0, host_ram)
        vram = _peak_vram_mb()
        if vram is not None:
            self._peak_vram_mb = max(self._peak_vram_mb or 0.0, vram)

    def snapshot(self) -> DataPackBuildPerformanceSnapshot:
        return DataPackBuildPerformanceSnapshot(
            peak_host_ram_mb=self._peak_host_ram_mb,
            peak_vram_mb=self._peak_vram_mb,
        )


def _host_ram_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _peak_vram_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)
