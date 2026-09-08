"""Provider-neutral ports for embedding diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.embedding_diagnostics.contracts import (
    BatchLatencyMeasurement,
    CudaEnvironmentSnapshot,
    EmbeddingDiagnosticReport,
)


class TokenCounterPort(Protocol):
    def count_tokens(self, text: str) -> int: ...


class EmbeddingProfilerPort(Protocol):
    def record_batch(
        self,
        *,
        batch_index: int,
        batch_size: int,
        record_count: int,
        input_tokens: int,
        batch_latency_seconds: float,
        inference_latency_seconds: float,
    ) -> BatchLatencyMeasurement: ...


class GpuTelemetryPort(Protocol):
    def environment(self) -> CudaEnvironmentSnapshot: ...

    def reset_peak_memory(self) -> None: ...

    def peak_memory_mb(self) -> float | None: ...

    def sample_utilization_percent(self) -> float | None: ...

    def average_utilization_percent(self) -> float | None: ...


class DiagnosticSinkPort(Protocol):
    def write_report(
        self,
        output_dir: Path,
        report: EmbeddingDiagnosticReport,
    ) -> tuple[Path, Path]: ...
