# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only execution artifact metadata for cross-domain consumers (artifact spine seam)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
)


class ExecutionArtifactLifecycleStatus(StrEnum):
    REGISTERED = "registered"


@dataclass(frozen=True, slots=True)
class ExecutionArtifactExecutionScope:
    tenant_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId | None
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class ExecutionArtifactMetadataRecord:
    """Safe artifact metadata — no binary payload or secret-bearing URIs."""

    artifact_ref: str
    artifact_type: str
    lifecycle_status: ExecutionArtifactLifecycleStatus
    content_classification: str
    execution_id: ExecutionId
    attempt_id: AttemptId | None
    sequence_key: int
    evidence_refs: tuple[str, ...]
    safe_summary: str


@dataclass(frozen=True, slots=True)
class ExecutionArtifactMetadataReadResult:
    records: tuple[ExecutionArtifactMetadataRecord, ...]
    is_truncated: bool


@runtime_checkable
class ExecutionArtifactMetadataReadPort(Protocol):
    """Lists artifact metadata registered for one execution scope — read only."""

    @property
    def source_id(self) -> str: ...

    def list_artifact_metadata(
        self,
        scope: ExecutionArtifactExecutionScope,
        *,
        limit: int,
    ) -> ExecutionArtifactMetadataReadResult: ...


__all__ = [
    "ExecutionArtifactExecutionScope",
    "ExecutionArtifactLifecycleStatus",
    "ExecutionArtifactMetadataReadPort",
    "ExecutionArtifactMetadataReadResult",
    "ExecutionArtifactMetadataRecord",
]
