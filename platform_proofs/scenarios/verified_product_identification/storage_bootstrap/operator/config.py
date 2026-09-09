"""Immutable operator configuration for VPI full Data Pack storage load."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchSize,
    RelationalTargetId,
    ResumeMode,
    VectorTargetId,
    VerificationMode,
)

OPERATOR_SCHEMA_VERSION = "vpi.storage_load_operator/1.0.0"
PRODUCTION_COMPOSITION = "postgresql+qdrant"


class OperatorRunMode(str, Enum):
    PLAN = "PLAN"
    FRESH = "FRESH"
    RESUME = "RESUME"


@dataclass(frozen=True, slots=True)
class StorageLoadOperatorConfig:
    artifact_root: Path
    checkpoint_root: Path
    evidence_root: Path
    relational_target: RelationalTargetId
    vector_target: VectorTargetId
    batch_size: BootstrapBatchSize
    run_mode: OperatorRunMode
    verification_mode: VerificationMode
    expected_record_count: int | None = None
    expected_data_pack_content_identity: str | None = None

    def __post_init__(self) -> None:
        if not str(self.relational_target).strip():
            raise ValueError("relational_target must be non-empty")
        if not str(self.vector_target).strip():
            raise ValueError("vector_target must be non-empty")
        if self.expected_record_count is not None and self.expected_record_count <= 0:
            raise ValueError("expected_record_count must be > 0 when set")
        if (
            self.expected_data_pack_content_identity is not None
            and not self.expected_data_pack_content_identity.strip()
        ):
            raise ValueError("expected_data_pack_content_identity must be non-empty when set")

    @property
    def plan_only(self) -> bool:
        return self.run_mode is OperatorRunMode.PLAN

    @property
    def resume_mode(self) -> ResumeMode:
        if self.run_mode is OperatorRunMode.RESUME:
            return ResumeMode.RESUME
        return ResumeMode.FRESH
