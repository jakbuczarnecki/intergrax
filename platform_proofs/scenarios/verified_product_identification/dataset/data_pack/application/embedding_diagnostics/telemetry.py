"""GPU telemetry implementations with graceful CPU-only fallback."""

from __future__ import annotations

import subprocess

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    CudaEnvironmentSnapshot,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.ports import (
    GpuTelemetryPort,
)


class NoOpGpuTelemetry(GpuTelemetryPort):
    """Safe fallback when CUDA or telemetry tooling is unavailable."""

    def environment(self) -> CudaEnvironmentSnapshot:
        return CudaEnvironmentSnapshot(cuda_available=False, gpu_name=None)

    def reset_peak_memory(self) -> None:
        return None

    def peak_memory_mb(self) -> float | None:
        return None

    def sample_utilization_percent(self) -> float | None:
        return None

    def average_utilization_percent(self) -> float | None:
        return None


class TorchGpuTelemetry(GpuTelemetryPort):
    """Collects CUDA memory stats via torch and utilization via nvidia-smi."""

    def __init__(self) -> None:
        self._utilization_samples: list[float] = []

    def environment(self) -> CudaEnvironmentSnapshot:
        try:
            import torch
        except ImportError:
            return CudaEnvironmentSnapshot(cuda_available=False, gpu_name=None)
        if not torch.cuda.is_available():
            return CudaEnvironmentSnapshot(cuda_available=False, gpu_name=None)
        return CudaEnvironmentSnapshot(
            cuda_available=True,
            gpu_name=torch.cuda.get_device_name(0),
        )

    def reset_peak_memory(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def peak_memory_mb(self) -> float | None:
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024.0 * 1024.0)

    def sample_utilization_percent(self) -> float | None:
        utilization = _query_gpu_utilization_percent()
        if utilization is not None:
            self._utilization_samples.append(utilization)
        return utilization

    def average_utilization_percent(self) -> float | None:
        if not self._utilization_samples:
            return None
        return sum(self._utilization_samples) / len(self._utilization_samples)


def create_gpu_telemetry() -> GpuTelemetryPort:
    environment = TorchGpuTelemetry().environment()
    if not environment.cuda_available:
        return NoOpGpuTelemetry()
    return TorchGpuTelemetry()


def _query_gpu_utilization_percent() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    first_line = completed.stdout.strip().splitlines()
    if not first_line:
        return None
    try:
        return float(first_line[0].strip())
    except ValueError:
        return None
