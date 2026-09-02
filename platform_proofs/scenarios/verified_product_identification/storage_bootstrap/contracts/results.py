"""Typed bootstrap validation and run reports — no bool-only outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.model import (
    BootstrapState,
    VpiBootstrapManifest,
)


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class ValidationCheck:
    name: str
    status: ValidationStatus
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    status: ValidationStatus
    checks: tuple[ValidationCheck, ...]

    @classmethod
    def from_checks(cls, checks: tuple[ValidationCheck, ...]) -> ValidationReport:
        overall = (
            ValidationStatus.PASS
            if all(check.status is ValidationStatus.PASS for check in checks)
            else ValidationStatus.FAIL
        )
        return cls(status=overall, checks=checks)


@dataclass(frozen=True, slots=True)
class EmbeddingProbeResult:
    status: ValidationStatus
    provider: str
    model: str
    resolved_dimension: int
    probe_vector_count: int
    detail: str


@dataclass(frozen=True, slots=True)
class BootstrapBatchProgress:
    batch_ordinal: int
    rows_in_batch: int
    cumulative_rows_processed: int
    catalog_rows_ingested: int
    search_points_ingested: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class BootstrapRunReport:
    final_state: BootstrapState
    manifest: VpiBootstrapManifest | None
    validation: ValidationReport | None
    embedding_probe: EmbeddingProbeResult | None
    batches_completed: int
    rows_processed: int
    failure_stage: str | None
    failure_detail: str | None
