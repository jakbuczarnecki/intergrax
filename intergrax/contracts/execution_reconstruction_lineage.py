# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Neutral lineage projection types for execution reconstruction read models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageSegmentLifecycle,
)


class ExecutionLineageReconstructionIntegrityError(Exception):
    """Raised when durable lineage fails forensic structural validation."""


class ExecutionLineageReadStatus(StrEnum):
    AVAILABLE = "available"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


class ExecutionLineageCompleteness(StrEnum):
    OPEN = "open"
    COMPLETE = "complete"
    PARTIAL = "partial"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class ReconstructedLineageSegment:
    root_execution_id: ExecutionId
    predecessor_root_execution_id: ExecutionId | None
    lifecycle: ExecutionLineageSegmentLifecycle
    admissions: tuple[ExecutionLineageAdmissionRecord, ...]


@dataclass(frozen=True, slots=True)
class ReconstructedAttemptLineage:
    attempt_id: AttemptId
    read_status: ExecutionLineageReadStatus
    completeness: ExecutionLineageCompleteness | None
    degraded: bool | None
    closure_kind: ExecutionLineageAttemptClosureKind | None
    segments: tuple[ReconstructedLineageSegment, ...]
    discovery_contract_version: int | None = None
    discovery_position: int | None = None


__all__ = [
    "ExecutionLineageCompleteness",
    "ExecutionLineageReadStatus",
    "ExecutionLineageReconstructionIntegrityError",
    "ReconstructedAttemptLineage",
    "ReconstructedLineageSegment",
]
