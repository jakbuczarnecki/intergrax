# © Artur Czarnecki. All rights reserved.

"""Memory observability contracts (MEM-ENT-12)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intergrax.contracts.execution_identity import EventId
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceOperation,
    MemoryGovernanceReasonCode,
)

__all__ = [
    "MemoryDiagnosticComponent",
    "MemoryDiagnosticCounts",
    "MemoryDiagnosticEvent",
    "MemoryDiagnosticFailureClass",
    "MemoryDiagnosticOperation",
    "MemoryDiagnosticOutcome",
    "MemoryDiagnosticPhase",
    "MemoryObservabilitySink",
    "NoOpMemoryObservabilitySink",
    "RecordingMemoryObservabilitySink",
]


class MemoryDiagnosticPhase(str, Enum):
    GOVERNANCE = "governance"
    TERMINAL = "terminal"
    PROJECTION = "projection"
    RECONCILIATION = "reconciliation"


class MemoryDiagnosticComponent(str, Enum):
    CONTROL_PLANE = "control_plane"
    GOVERNANCE = "governance"
    LIFECYCLE = "lifecycle"
    ENTITY_TEMPORAL = "entity_temporal"
    PROCEDURAL = "procedural"
    LONG_HORIZON = "long_horizon"
    RECONCILIATION = "reconciliation"


class MemoryDiagnosticOperation(str, Enum):
    REMEMBER = "remember"
    RECALL = "recall"
    FORGET = "forget"
    DELETE = "delete"
    SUPERSEDE = "supersede"
    UPDATE = "update"
    PROMOTE = "promote"
    COMPACT = "compact"
    PROJECT = "project"
    GOVERNANCE_EVALUATE = "governance_evaluate"
    PROJECTION_WRITE = "projection_write"
    PROJECTION_DELETE = "projection_delete"
    RECONCILE = "reconcile"
    LINEAGE_TRAVERSE = "lineage_traverse"
    SOURCE_RESOLVE = "source_resolve"
    PROVIDER_QUALIFICATION = "provider_qualification"


class MemoryDiagnosticOutcome(str, Enum):
    SUCCESS = "success"
    DENIED = "denied"
    REVIEW_REQUIRED = "review_required"
    NOT_FOUND = "not_found"
    PARTIAL = "partial"
    FAILED = "failed"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"


class MemoryDiagnosticFailureClass(str, Enum):
    POLICY = "policy"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    SOURCE_AUTHORITY = "source_authority"
    STORE = "store"
    PROVIDER = "provider"
    CONCURRENCY = "concurrency"
    RECONCILIATION = "reconciliation"
    PROJECTION = "projection"
    TIMEOUT = "timeout"
    UNSUPPORTED = "unsupported"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class MemoryDiagnosticCounts:
    sources_requested: int | None = None
    sources_resolved: int | None = None
    summaries_created: int | None = None
    failures: int | None = None


@dataclass(frozen=True, slots=True)
class MemoryDiagnosticEvent:
    """Vendor-neutral memory operation diagnostic (no raw memory content)."""

    event_id: EventId
    operation: MemoryDiagnosticOperation
    phase: MemoryDiagnosticPhase
    outcome: MemoryDiagnosticOutcome
    component: MemoryDiagnosticComponent
    reference_time_iso: str
    tenant_id: str | None = None
    user_id: str | None = None
    workspace_id: str | None = None
    memory_id: str | None = None
    revision: int | None = None
    governance_operation: MemoryGovernanceOperation | None = None
    provider_id: str | None = None
    projection_id: str | None = None
    policy_id: str | None = None
    reason_code: MemoryGovernanceReasonCode | None = None
    failure_class: MemoryDiagnosticFailureClass | None = None
    duration_seconds: float | None = None
    counts: MemoryDiagnosticCounts | None = None


class MemoryObservabilitySink(Protocol):
    def record(self, event: MemoryDiagnosticEvent) -> None:
        """Receive a diagnostic event; must not mutate memory state."""


class NoOpMemoryObservabilitySink:
    def record(self, event: MemoryDiagnosticEvent) -> None:
        return None


class RecordingMemoryObservabilitySink:
    """Test double collecting emitted diagnostics."""

    def __init__(self) -> None:
        self.events: list[MemoryDiagnosticEvent] = []

    def record(self, event: MemoryDiagnosticEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()
