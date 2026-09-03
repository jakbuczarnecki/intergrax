"""Typed materialization validation and run reports."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EmbeddingArtifactManifest,
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class MaterializationRunReport:
    final_state: EmbeddingArtifactState
    manifest: EmbeddingArtifactManifest | None
    validation: ValidationReport | None
    embedding_probe: EmbeddingProbeResult | None
    rows_materialized: int
    embedding_batches: int
    shards_committed: int
    embedding_calls: int
    elapsed_total_seconds: float
    elapsed_embedding_seconds: float
    elapsed_derive_seconds: float
    elapsed_artifact_write_seconds: float
    failure_stage: str | None
    failure_detail: str | None

    @property
    def effective_records_per_second(self) -> float:
        if self.elapsed_total_seconds <= 0:
            return 0.0
        return self.rows_materialized / self.elapsed_total_seconds
