# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness event catalog — single source of truth for spine metadata (OBS-EVOL-9.1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.contracts.spine_event_metadata import (
    category_for_spine_type,
    spine_ops_filter_hint_for_type,
)
from intergrax.runtime.events.event_taxonomy import (
    EventCategory,
    RetentionClass,
    category_for_event_kind,
)

OpsFilterHint = str

# High-volume spine types sampled before persist (OBS-EVOL-9.6 enforces at bus).
_DEFAULT_SAMPLE_RATE = 1.0
_SAMPLED_EVENT_TYPES: dict[RuntimeEventType, float] = {
    RuntimeEventType.TASK_PROGRESS: 0.25,
}


_AUDIT_RETENTION_PREFIXES = (
    "TOOL_",
    "HUMAN_",
    "POLICY_",
    "BUDGET_",
    "INTERRUPT_",
)
_DEBUG_RETENTION_TYPES = frozenset({RuntimeEventType.TASK_PROGRESS})


@dataclass(frozen=True, slots=True)
class EventCatalogEntry:
    """Metadata for one spine ``RuntimeEventType``."""

    event_type: RuntimeEventType
    phase: ExecutionPhase
    ops_hint: OpsFilterHint
    category: EventCategory
    preferred_payload_schema_id: str | None = None
    sample_rate: float = _DEFAULT_SAMPLE_RATE
    retention_class: RetentionClass = RetentionClass.OPERATIONAL
    consolidation_kind: str | None = None

    @property
    def default_event_kind(self) -> str:
        return self.event_type.value


# Canonical phase + ops maps (migrated from phase_coverage.py).
_SPINE_PHASE: dict[RuntimeEventType, ExecutionPhase] = {
    RuntimeEventType.TASK_CREATED: ExecutionPhase.INTAKE,
    RuntimeEventType.TASK_CLASSIFIED: ExecutionPhase.CLASSIFICATION,
    RuntimeEventType.PLAN_CREATED: ExecutionPhase.PLANNING,
    RuntimeEventType.PLAN_UPDATED: ExecutionPhase.PLANNING,
    RuntimeEventType.PLAN_FAILED: ExecutionPhase.PLANNING,
    RuntimeEventType.AGENT_SELECTED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.CONTEXT_BUILT: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_ASSEMBLED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_CANDIDATE_DROPPED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_VALIDATION_FAILED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.CONTEXT_TRIMMED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.INGESTION_FAILED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.SKILL_RESOLVED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.SKILL_IMPORT_FAILED: ExecutionPhase.AGENT_SELECTION,
    RuntimeEventType.STEP_STARTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.STEP_COMPLETED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.STEP_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.EXECUTION_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.EXTERNAL_OPERATION_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_REQUESTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_COMPLETED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_DENIED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TOOL_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.VALIDATION_STARTED: ExecutionPhase.VALIDATION,
    RuntimeEventType.VALIDATION_PASSED: ExecutionPhase.VALIDATION,
    RuntimeEventType.VALIDATION_FAILED: ExecutionPhase.VALIDATION,
    RuntimeEventType.DECISION_EMITTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.INTERRUPT_REQUESTED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.INTERRUPT_HANDLED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.INTERRUPT_ESCALATED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.HUMAN_APPROVAL_TIMEOUT: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.PAUSE_REQUESTED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.PAUSED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.RESUMED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.RETRY_SCHEDULED: ExecutionPhase.RETRY_HANDLING,
    RuntimeEventType.RETRY_STARTED: ExecutionPhase.RETRY_HANDLING,
    RuntimeEventType.CANCELLATION_REQUESTED: ExecutionPhase.COMPLETION,
    RuntimeEventType.CANCELLED: ExecutionPhase.COMPLETION,
    RuntimeEventType.MEMORY_READ: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.MEMORY_WRITE: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.HANDOFF_INITIATED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.HANDOFF_COMPLETED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.DELEGATION_GRANTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TRACE_PERSISTED: ExecutionPhase.TRACE_PERSISTENCE,
    RuntimeEventType.TASK_PROGRESS: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.TASK_COMPLETED: ExecutionPhase.COMPLETION,
    RuntimeEventType.TASK_FAILED: ExecutionPhase.COMPLETION,
    RuntimeEventType.RUNTIME_HANDLER_FAILED: ExecutionPhase.INTERRUPT_HANDLING,
    RuntimeEventType.LLM_CALL: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.POLICY_DECISION: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.GRAPH_BACKPRESSURE: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.GUARDRAIL_BLOCKED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.BUDGET_THRESHOLD: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.BUDGET_EXCEEDED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.DOMAIN_SIGNAL: ExecutionPhase.STEP_EXECUTION,
}


def consolidation_kind_for_spine_type(event_type: RuntimeEventType) -> str | None:
    """Deprecated — consolidated types removed from spine (OBS-EVOL-9.7)."""
    return None


def retention_class_for_spine_type(event_type: RuntimeEventType) -> RetentionClass:
    if event_type in _DEBUG_RETENTION_TYPES:
        return RetentionClass.DEBUG
    name = event_type.name
    if any(name.startswith(prefix) for prefix in _AUDIT_RETENTION_PREFIXES):
        return RetentionClass.AUDIT
    if name.startswith("VALIDATION_") or name == "LLM_CALL":
        return RetentionClass.AUDIT
    return RetentionClass.OPERATIONAL


def sample_rate_for_spine_type(event_type: RuntimeEventType) -> float:
    return _SAMPLED_EVENT_TYPES.get(event_type, _DEFAULT_SAMPLE_RATE)


def build_event_catalog() -> dict[RuntimeEventType, EventCatalogEntry]:
    from intergrax.runtime.events.payload_registry import EVENT_TYPE_PREFERRED_SCHEMA

    catalog: dict[RuntimeEventType, EventCatalogEntry] = {}
    for event_type in RuntimeEventType:
        phase = _SPINE_PHASE.get(event_type)
        ops_hint = spine_ops_filter_hint_for_type(event_type)
        if phase is None or ops_hint is None:
            continue
        catalog[event_type] = EventCatalogEntry(
            event_type=event_type,
            phase=phase,
            ops_hint=ops_hint,
            category=category_for_spine_type(event_type),
            preferred_payload_schema_id=EVENT_TYPE_PREFERRED_SCHEMA.get(event_type),
            sample_rate=sample_rate_for_spine_type(event_type),
            retention_class=retention_class_for_spine_type(event_type),
            consolidation_kind=consolidation_kind_for_spine_type(event_type),
        )
    return catalog


EVENT_CATALOG: Final[dict[RuntimeEventType, EventCatalogEntry]] = build_event_catalog()

# Backward-compatible views (phase_coverage.py re-exports these).
EVENT_PHASE_COVERAGE: Final[dict[RuntimeEventType, ExecutionPhase]] = {
    event_type: entry.phase for event_type, entry in EVENT_CATALOG.items()
}
EVENT_OPS_FILTER_HINTS: Final[dict[RuntimeEventType, OpsFilterHint]] = {
    event_type: entry.ops_hint for event_type, entry in EVENT_CATALOG.items()
}


def get_catalog_entry(event_type: RuntimeEventType) -> EventCatalogEntry | None:
    return EVENT_CATALOG.get(event_type)


def phase_for_event(event_type: RuntimeEventType) -> ExecutionPhase | None:
    entry = get_catalog_entry(event_type)
    return entry.phase if entry is not None else None


def ops_filter_hint_for_event(event_type: RuntimeEventType) -> OpsFilterHint | None:
    entry = get_catalog_entry(event_type)
    return entry.ops_hint if entry is not None else None


def list_unmapped_event_types() -> list[RuntimeEventType]:
    return [item for item in RuntimeEventType if item not in EVENT_CATALOG]


def list_unmapped_ops_filter_hints() -> list[RuntimeEventType]:
    return list_unmapped_event_types()


def should_persist_event(event: RuntimeEvent) -> bool:
    """Deterministic sampling gate for spine and platform domain signals (OBS-EVOL-9.6/9.7)."""
    from intergrax.runtime.events.spine_consolidation import (
        should_persist_platform_kind,
    )

    if (
        event.event_type == RuntimeEventType.DOMAIN_SIGNAL
        and event.event_kind.startswith("platform.")
    ):
        return should_persist_platform_kind(event.event_kind, event.event_id)
    entry = get_catalog_entry(event.event_type)
    if entry is None or entry.sample_rate >= 1.0:
        return True
    digest = int(hashlib.sha256(event.event_id.encode("utf-8")).hexdigest()[:8], 16)
    return (digest % 10_000) < int(entry.sample_rate * 10_000)


__all__ = [
    "EVENT_CATALOG",
    "EVENT_OPS_FILTER_HINTS",
    "EVENT_PHASE_COVERAGE",
    "EventCatalogEntry",
    "OpsFilterHint",
    "category_for_event_kind",
    "category_for_spine_type",
    "consolidation_kind_for_spine_type",
    "get_catalog_entry",
    "list_unmapped_event_types",
    "list_unmapped_ops_filter_hints",
    "ops_filter_hint_for_event",
    "phase_for_event",
    "retention_class_for_spine_type",
    "sample_rate_for_spine_type",
    "should_persist_event",
]
