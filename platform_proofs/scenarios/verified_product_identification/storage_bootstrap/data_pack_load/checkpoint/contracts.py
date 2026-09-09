"""Typed durable checkpoint contracts for storage bootstrap resume."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    RelationalTargetId,
    VectorTargetId,
    VerificationMode,
)

VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION = "vpi-storage-bootstrap-checkpoint-v1"
VPI_STORAGE_BOOTSTRAP_ORDERING_POLICY = "global_row_index_asc_v1"


@dataclass(frozen=True, slots=True)
class BootstrapRunIdentity:
    """Content-bound bootstrap run identity — location alone is insufficient."""

    data_pack_content_identity: str
    data_pack_version: str
    record_count: int
    batch_size: int
    relational_target: RelationalTargetId
    vector_target: VectorTargetId
    verification_mode: VerificationMode
    ordering_policy: str
    state_schema_version: str
    source_dataset_sha256: str
    embedding_model_identity: str

    def __post_init__(self) -> None:
        if not self.data_pack_content_identity.strip():
            raise ValueError("data_pack_content_identity must be non-empty")
        if self.record_count < 0:
            raise ValueError("record_count must be >= 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.ordering_policy != VPI_STORAGE_BOOTSTRAP_ORDERING_POLICY:
            raise ValueError(f"unsupported ordering_policy: {self.ordering_policy}")


@dataclass(frozen=True, slots=True)
class BootstrapCheckpointIdentity:
    """Stable checkpoint namespace derived from run identity digest."""

    run_identity_digest: str

    def __post_init__(self) -> None:
        if len(self.run_identity_digest) != 64:
            raise ValueError("run_identity_digest must be a 64-char sha256 hex digest")


@dataclass(frozen=True, slots=True)
class BootstrapBatchCheckpoint:
    batch_number: int
    phase: BootstrapBatchPhase
    last_global_row_index: int
    last_identity: str

    def __post_init__(self) -> None:
        if self.batch_number < 0:
            raise ValueError("batch_number must be >= 0")
        if self.last_global_row_index < 0:
            raise ValueError("last_global_row_index must be >= 0")
        if not self.last_identity.strip():
            raise ValueError("last_identity must be non-empty")


@dataclass(frozen=True, slots=True)
class BootstrapCheckpointState:
    schema_version: str
    run_identity: BootstrapRunIdentity
    batch_size: int
    total_records: int
    batch_count: int
    last_committed_batch_number: int | None
    last_committed_global_row_index: int | None
    last_committed_identity: str | None
    committed_record_count: int
    batch_phases: tuple[BootstrapBatchPhase, ...]
    state_revision: int
    updated_at_utc: str
    current_batch_phase: BootstrapBatchPhase | None = None

    def __post_init__(self) -> None:
        if self.schema_version != VPI_STORAGE_BOOTSTRAP_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.total_records < 0:
            raise ValueError("total_records must be >= 0")
        if self.batch_count < 0:
            raise ValueError("batch_count must be >= 0")
        if self.committed_record_count < 0:
            raise ValueError("committed_record_count must be >= 0")
        if self.state_revision < 1:
            raise ValueError("state_revision must be >= 1")
        if len(self.batch_phases) != self.batch_count:
            raise ValueError("batch_phases length must equal batch_count")
        if self.last_committed_batch_number is None:
            if self.committed_record_count != 0:
                raise ValueError("committed_record_count must be 0 when no batch committed")
        elif self.last_committed_batch_number < 0:
            raise ValueError("last_committed_batch_number must be >= 0")


@dataclass(frozen=True, slots=True)
class BootstrapCheckpointCompatibility:
    is_compatible: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class BootstrapResumeDecision:
    start_batch_number: int
    committed_batch_count: int
    committed_record_count: int

    def __post_init__(self) -> None:
        if self.start_batch_number < 0:
            raise ValueError("start_batch_number must be >= 0")
        if self.committed_batch_count < 0:
            raise ValueError("committed_batch_count must be >= 0")
        if self.committed_record_count < 0:
            raise ValueError("committed_record_count must be >= 0")
