# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral factual execution reconstruction read models (OBS-CONTRACT-BOUNDARY-1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, RunId, TaskId
from intergrax.contracts.execution_reconstruction_lineage import (
    ExecutionLineageCompleteness,
    ExecutionLineageReadStatus,
    ReconstructedAttemptLineage,
)
from intergrax.contracts.platform_causal_evidence import PlatformCausalEvidence
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent


class ExecutionReconstructionIntegrityError(Exception):
    """Raised when canonical persistence returns facts outside the requested scope."""


class RuntimeHistoryCompleteness(StrEnum):
    """Whether positioned runtime history for the run is complete or truncated."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"


class ExecutionAttemptDiscoveryReadStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class ExecutionAttemptDiscoveryCompleteness(StrEnum):
    COMPLETE = "complete"
    LEGACY_UNKNOWN = "legacy_unknown"
    TRUNCATED = "truncated"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ReconstructedAttempt:
    """One attempt within an execution reconstruction — derived, not canonical."""

    attempt_id: AttemptId
    causal_evidence: tuple[PlatformCausalEvidence, ...]
    positioned_events: tuple[PositionedRuntimeEvent, ...]
    lineage: ReconstructedAttemptLineage | None = None

    @property
    def has_transport_evidence(self) -> bool:
        return bool(self.causal_evidence)

    @property
    def has_runtime_events(self) -> bool:
        return bool(self.positioned_events)


@dataclass(frozen=True, slots=True)
class ExecutionReconstruction:
    """
    Derived read model joining runtime execution history and causal evidence.

    NOT persisted and NOT a source of truth.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    causal_evidence: tuple[PlatformCausalEvidence, ...]
    positioned_events: tuple[PositionedRuntimeEvent, ...]
    attempts: tuple[ReconstructedAttempt, ...]
    runtime_history_completeness: RuntimeHistoryCompleteness
    attempt_discovery_read_status: ExecutionAttemptDiscoveryReadStatus | None = None
    attempt_discovery_completeness: ExecutionAttemptDiscoveryCompleteness | None = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def has_transport_evidence(self) -> bool:
        return bool(self.causal_evidence)

    @property
    def has_runtime_events(self) -> bool:
        return bool(self.positioned_events)

    @property
    def is_runtime_history_complete(self) -> bool:
        return self.runtime_history_completeness is RuntimeHistoryCompleteness.COMPLETE

    @property
    def has_lineage_evidence(self) -> bool:
        return any(
            attempt.lineage is not None
            and attempt.lineage.read_status is ExecutionLineageReadStatus.AVAILABLE
            for attempt in self.attempts
        )

    @property
    def has_complete_lineage(self) -> bool:
        return any(
            attempt.lineage is not None
            and attempt.lineage.completeness is ExecutionLineageCompleteness.COMPLETE
            for attempt in self.attempts
        )

    @property
    def has_partial_lineage(self) -> bool:
        return any(
            attempt.lineage is not None
            and attempt.lineage.completeness is ExecutionLineageCompleteness.PARTIAL
            for attempt in self.attempts
        )


__all__ = [
    "ExecutionAttemptDiscoveryCompleteness",
    "ExecutionAttemptDiscoveryReadStatus",
    "ExecutionReconstruction",
    "ExecutionReconstructionIntegrityError",
    "ReconstructedAttempt",
    "RuntimeHistoryCompleteness",
]
