"""Immutable provider-neutral contracts for Data Pack storage bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NewType

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    BootstrapFailure,
)

RelationalTargetId = NewType("RelationalTargetId", str)
VectorTargetId = NewType("VectorTargetId", str)


class ResumeMode(str, Enum):
    FRESH = "FRESH"
    RESUME = "RESUME"


class VerificationMode(str, Enum):
    STRICT = "STRICT"
    SKIP = "SKIP"


class BootstrapFinalStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class BootstrapBatchPhase(str, Enum):
    PENDING = "PENDING"
    RELATIONAL_WRITING = "RELATIONAL_WRITING"
    VECTOR_WRITING = "VECTOR_WRITING"
    VERIFYING = "VERIFYING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class BootstrapBatchSize:
    value: int

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("batch_size must be > 0")


@dataclass(frozen=True, slots=True)
class BootstrapRequest:
    artifact_root: Path
    relational_target: RelationalTargetId
    vector_target: VectorTargetId
    batch_size: BootstrapBatchSize
    resume_mode: ResumeMode = ResumeMode.FRESH
    verification_mode: VerificationMode = VerificationMode.STRICT
    plan_only: bool = False

    def __post_init__(self) -> None:
        if not str(self.relational_target).strip():
            raise ValueError("relational_target must be non-empty")
        if not str(self.vector_target).strip():
            raise ValueError("vector_target must be non-empty")


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    record_count: int
    batch_size: int
    batch_count: int
    final_batch_size: int
    relational_target: RelationalTargetId
    vector_target: VectorTargetId

    def __post_init__(self) -> None:
        if self.record_count < 0:
            raise ValueError("record_count must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.batch_count < 0:
            raise ValueError("batch_count must be >= 0")
        if self.final_batch_size < 0:
            raise ValueError("final_batch_size must be >= 0")


@dataclass(frozen=True, slots=True)
class RelationalLoadRecord:
    source_ref: SourceRecordRef
    global_row_index: int
    record_json: str
    semantic_text: str
    semantic_text_hash: str
    derivation_version: str

    def __post_init__(self) -> None:
        if self.global_row_index < 0:
            raise ValueError("global_row_index must be >= 0")
        if not self.record_json.strip():
            raise ValueError("record_json must be non-empty")
        if not self.semantic_text_hash.strip():
            raise ValueError("semantic_text_hash must be non-empty")
        if not self.derivation_version.strip():
            raise ValueError("derivation_version must be non-empty")


@dataclass(frozen=True, slots=True)
class VectorLoadRecord:
    logical_point_id: str
    source_ref: SourceRecordRef
    semantic_text_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str | None
    embedding_dimension: int
    dense_embedding: tuple[float, ...]
    derivation_version: str

    def __post_init__(self) -> None:
        if not self.logical_point_id.strip():
            raise ValueError("logical_point_id must be non-empty")
        if not self.semantic_text_hash.strip():
            raise ValueError("semantic_text_hash must be non-empty")
        if not self.embedding_provider.strip():
            raise ValueError("embedding_provider must be non-empty")
        if not self.embedding_model.strip():
            raise ValueError("embedding_model must be non-empty")
        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be > 0")
        if len(self.dense_embedding) != self.embedding_dimension:
            msg = (
                f"dense_embedding length {len(self.dense_embedding)} "
                f"!= embedding_dimension {self.embedding_dimension}"
            )
            raise ValueError(msg)
        if not self.derivation_version.strip():
            raise ValueError("derivation_version must be non-empty")


@dataclass(frozen=True, slots=True)
class RelationalBatch:
    batch_number: int
    target: RelationalTargetId
    records: tuple[RelationalLoadRecord, ...]

    def __post_init__(self) -> None:
        if self.batch_number < 0:
            raise ValueError("batch_number must be >= 0")


@dataclass(frozen=True, slots=True)
class VectorBatch:
    batch_number: int
    target: VectorTargetId
    records: tuple[VectorLoadRecord, ...]

    def __post_init__(self) -> None:
        if self.batch_number < 0:
            raise ValueError("batch_number must be >= 0")


@dataclass(frozen=True, slots=True)
class StorageLoadBatchResult:
    requested_count: int
    written_count: int
    updated_count: int
    skipped_count: int
    failed_count: int
    first_failed_identity: str | None = None

    def __post_init__(self) -> None:
        if self.requested_count < 0:
            raise ValueError("requested_count must be >= 0")
        if self.written_count < 0:
            raise ValueError("written_count must be >= 0")
        if self.updated_count < 0:
            raise ValueError("updated_count must be >= 0")
        if self.skipped_count < 0:
            raise ValueError("skipped_count must be >= 0")
        if self.failed_count < 0:
            raise ValueError("failed_count must be >= 0")
        applied = self.written_count + self.updated_count + self.skipped_count + self.failed_count
        if applied != self.requested_count:
            msg = (
                f"result counts must sum to requested_count "
                f"({applied} != {self.requested_count})"
            )
            raise ValueError(msg)

    @property
    def successful_count(self) -> int:
        return self.written_count + self.updated_count + self.skipped_count

    @property
    def is_complete_success(self) -> bool:
        return self.failed_count == 0 and self.successful_count == self.requested_count


@dataclass(frozen=True, slots=True)
class BootstrapProgress:
    phase: BootstrapBatchPhase
    batch_number: int
    records_processed: int
    total_records: int
    elapsed_seconds: float
    last_identity: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    status: BootstrapFinalStatus
    plan: BootstrapPlan
    total_expected_records: int
    total_relational_written: int
    total_vectors_written: int
    committed_batches: int
    failed_batches: int
    last_committed_global_row_index: int | None
    failure: BootstrapFailure | None = None
