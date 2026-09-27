# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only runtime fact inputs for intelligence context projection (W6-C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.runtime_intelligence.context import (
    RuntimeIntelligenceFactReference,
)


class RuntimeIntelligenceSignalKind(StrEnum):
    """Bounded signal families observed from canonical runtime stores."""

    EXECUTION_INSTABILITY = "execution_instability"
    REPEATED_FAILURES = "repeated_failures"
    RETRY_PRESSURE = "retry_pressure"
    RECOVERY_SIGNAL = "recovery_signal"
    RESOURCE_PRESSURE = "resource_pressure"


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceObservedSignal:
    kind: RuntimeIntelligenceSignalKind
    intensity: float
    source_fact_ref: str

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity <= 1.0):
            raise ValueError("intensity must be in [0.0, 1.0]")
        if not self.source_fact_ref.strip():
            raise ValueError("source_fact_ref must be non-empty")


@dataclass(frozen=True, slots=True)
class RuntimeIntelligenceFacts:
    """
    Request-scoped snapshot assembled by integration layer (read ports only).

    Ownership: caller holds facts until context is built; no shared mutable state.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    fact_references: tuple[RuntimeIntelligenceFactReference, ...]
    correlation_id: str
    collected_at: datetime
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    observed_signals: tuple[RuntimeIntelligenceObservedSignal, ...] = ()
    context_label: str = ""

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        object.__setattr__(self, "task_id", validate_task_id(self.task_id))
        object.__setattr__(self, "run_id", validate_run_id(self.run_id))
        if self.attempt_id is not None:
            object.__setattr__(self, "attempt_id", validate_attempt_id(self.attempt_id))
        if self.execution_id is not None:
            object.__setattr__(
                self,
                "execution_id",
                validate_execution_id(self.execution_id),
            )
        if not self.correlation_id.strip():
            raise ValueError("correlation_id must be non-empty")
        if self.collected_at.tzinfo is None:
            raise ValueError("collected_at must be timezone-aware")


__all__ = [
    "RuntimeIntelligenceFacts",
    "RuntimeIntelligenceObservedSignal",
    "RuntimeIntelligenceSignalKind",
]
