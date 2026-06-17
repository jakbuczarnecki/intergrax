# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness event catalog — single source of truth for spine metadata (OBS-EVOL-9.1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.payload_registry import EVENT_TYPE_PREFERRED_SCHEMA
from intergrax.runtime.events.runtime_event import RuntimeEventType

OpsFilterHint = str

# High-volume spine types sampled before persist (OBS-EVOL-9.6 enforces at bus).
_DEFAULT_SAMPLE_RATE = 1.0
_SAMPLED_EVENT_TYPES: dict[RuntimeEventType, float] = {
    RuntimeEventType.TASK_PROGRESS: 0.25,
    RuntimeEventType.CAPACITY_SIGNAL_COLLECTED: 0.5,
}


class EventCategory(str, Enum):
    """Derived ops grouping for subscribers and metrics cardinality control."""

    TASK = "task"
    PLAN = "plan"
    TOOL = "tool"
    AGENT = "agent"
    CONTEXT = "context"
    HUMAN = "human"
    POLICY = "policy"
    PLATFORM = "platform"


class RetentionClass(str, Enum):
    """Store retention tier aligned with data classification (IDEAL-23.5)."""

    OPERATIONAL = "operational"
    AUDIT = "audit"
    DEBUG = "debug"


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
    RuntimeEventType.ADAPTIVE_SIGNAL_RECORDED: ExecutionPhase.TRACE_PERSISTENCE,
    RuntimeEventType.ADAPTIVE_PROPOSAL_SUBMITTED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_PROFILE_APPLIED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_PROFILE_ROLLBACK: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_VERIFICATION_FAILED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.ADAPTIVE_LOOP_BLOCKED: ExecutionPhase.FINALIZATION,
    RuntimeEventType.LLM_CALL: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.POLICY_DECISION: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.GRAPH_BACKPRESSURE: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.CAPACITY_SIGNAL_COLLECTED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.SCALE_EVALUATED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.SCALE_REQUESTED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.SCALE_APPROVED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.SCALE_DENIED: ExecutionPhase.HUMAN_APPROVAL,
    RuntimeEventType.SCALE_APPLIED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.SCALE_FAILED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.AUTONOMY_LEVEL_SET: ExecutionPhase.INTAKE,
    RuntimeEventType.AUTONOMY_LEVEL_CHANGED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.RECOVERY_REBOOT: ExecutionPhase.RETRY_HANDLING,
    RuntimeEventType.GUARDRAIL_BLOCKED: ExecutionPhase.CONTEXT_BUILDING,
    RuntimeEventType.BUDGET_THRESHOLD: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.BUDGET_EXCEEDED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.HOOK_BLOCKED: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.HOOK_ERROR: ExecutionPhase.STEP_EXECUTION,
    RuntimeEventType.HOOK_TIMEOUT: ExecutionPhase.STEP_EXECUTION,
}

_SPINE_OPS_HINT: dict[RuntimeEventType, OpsFilterHint] = {
    RuntimeEventType.TASK_CREATED: "trace:intake",
    RuntimeEventType.TASK_CLASSIFIED: "trace:classification",
    RuntimeEventType.PLAN_CREATED: "ops:planning",
    RuntimeEventType.PLAN_UPDATED: "ops:planning",
    RuntimeEventType.PLAN_FAILED: "ops:alert",
    RuntimeEventType.AGENT_SELECTED: "trace:selection",
    RuntimeEventType.CONTEXT_BUILT: "trace:context",
    RuntimeEventType.CONTEXT_ASSEMBLED: "trace:context",
    RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED: "trace:context",
    RuntimeEventType.CONTEXT_CANDIDATE_DROPPED: "trace:context",
    RuntimeEventType.CONTEXT_VALIDATION_FAILED: "ops:alert",
    RuntimeEventType.CONTEXT_TRIMMED: "trace:context",
    RuntimeEventType.INGESTION_FAILED: "ops:alert",
    RuntimeEventType.SKILL_RESOLVED: "trace:skills",
    RuntimeEventType.SKILL_IMPORT_FAILED: "ops:alert",
    RuntimeEventType.STEP_STARTED: "trace:step",
    RuntimeEventType.STEP_COMPLETED: "trace:step",
    RuntimeEventType.STEP_FAILED: "ops:alert",
    RuntimeEventType.TOOL_REQUESTED: "ops:tool_audit",
    RuntimeEventType.TOOL_COMPLETED: "ops:tool_audit",
    RuntimeEventType.TOOL_DENIED: "ops:alert",
    RuntimeEventType.TOOL_FAILED: "ops:alert",
    RuntimeEventType.VALIDATION_STARTED: "trace:validation",
    RuntimeEventType.VALIDATION_PASSED: "trace:validation",
    RuntimeEventType.VALIDATION_FAILED: "ops:alert",
    RuntimeEventType.DECISION_EMITTED: "trace:decision",
    RuntimeEventType.INTERRUPT_REQUESTED: "ops:hitl",
    RuntimeEventType.INTERRUPT_HANDLED: "ops:hitl",
    RuntimeEventType.INTERRUPT_ESCALATED: "ops:alert",
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: "ops:hitl",
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: "ops:hitl",
    RuntimeEventType.HUMAN_APPROVAL_TIMEOUT: "ops:alert",
    RuntimeEventType.PAUSE_REQUESTED: "ops:hitl",
    RuntimeEventType.PAUSED: "ops:hitl",
    RuntimeEventType.RESUMED: "ops:hitl",
    RuntimeEventType.RETRY_SCHEDULED: "ops:retry",
    RuntimeEventType.RETRY_STARTED: "ops:retry",
    RuntimeEventType.CANCELLATION_REQUESTED: "ops:completion",
    RuntimeEventType.CANCELLED: "ops:completion",
    RuntimeEventType.MEMORY_READ: "ops:memory",
    RuntimeEventType.MEMORY_WRITE: "ops:memory",
    RuntimeEventType.HANDOFF_INITIATED: "ops:handoff",
    RuntimeEventType.HANDOFF_COMPLETED: "ops:handoff",
    RuntimeEventType.DELEGATION_GRANTED: "ops:delegation",
    RuntimeEventType.TRACE_PERSISTED: "trace:persistence",
    RuntimeEventType.TASK_PROGRESS: "ops:progress",
    RuntimeEventType.TASK_COMPLETED: "ops:completion",
    RuntimeEventType.TASK_FAILED: "ops:alert",
    RuntimeEventType.RUNTIME_HANDLER_FAILED: "ops:alert",
    RuntimeEventType.ADAPTIVE_SIGNAL_RECORDED: "ops:adaptive",
    RuntimeEventType.ADAPTIVE_PROPOSAL_SUBMITTED: "ops:adaptive",
    RuntimeEventType.ADAPTIVE_PROFILE_APPLIED: "ops:adaptive",
    RuntimeEventType.ADAPTIVE_PROFILE_ROLLBACK: "ops:alert",
    RuntimeEventType.ADAPTIVE_VERIFICATION_FAILED: "ops:alert",
    RuntimeEventType.ADAPTIVE_LOOP_BLOCKED: "ops:alert",
    RuntimeEventType.LLM_CALL: "ops:llm_audit",
    RuntimeEventType.POLICY_DECISION: "ops:policy_audit",
    RuntimeEventType.GRAPH_BACKPRESSURE: "ops:backpressure",
    RuntimeEventType.CAPACITY_SIGNAL_COLLECTED: "ops:capacity",
    RuntimeEventType.SCALE_EVALUATED: "ops:capacity",
    RuntimeEventType.SCALE_REQUESTED: "ops:capacity",
    RuntimeEventType.SCALE_APPROVED: "ops:hitl",
    RuntimeEventType.SCALE_DENIED: "ops:hitl",
    RuntimeEventType.SCALE_APPLIED: "ops:capacity",
    RuntimeEventType.SCALE_FAILED: "ops:alert",
    RuntimeEventType.AUTONOMY_LEVEL_SET: "ops:governance",
    RuntimeEventType.AUTONOMY_LEVEL_CHANGED: "ops:governance",
    RuntimeEventType.RECOVERY_REBOOT: "ops:retry",
    RuntimeEventType.GUARDRAIL_BLOCKED: "ops:alert",
    RuntimeEventType.BUDGET_THRESHOLD: "ops:budget",
    RuntimeEventType.BUDGET_EXCEEDED: "ops:alert",
    RuntimeEventType.HOOK_BLOCKED: "ops:alert",
    RuntimeEventType.HOOK_ERROR: "ops:alert",
    RuntimeEventType.HOOK_TIMEOUT: "ops:alert",
}


def category_for_spine_type(event_type: RuntimeEventType) -> EventCategory:
    """Derive ``EventCategory`` from spine enum member name."""
    name = event_type.name
    if name.startswith("TASK_"):
        return EventCategory.TASK
    if name.startswith("PLAN_"):
        return EventCategory.PLAN
    if name.startswith("TOOL_"):
        return EventCategory.TOOL
    if name.startswith("STEP_") or name == "AGENT_SELECTED":
        return EventCategory.AGENT
    if name.startswith(("CONTEXT_", "MEMORY_", "SKILL_", "INGESTION_")):
        return EventCategory.CONTEXT
    if name.startswith(("HUMAN_", "PAUSE_", "INTERRUPT_", "RETRY_")) or name in {
        "PAUSED",
        "RESUMED",
    }:
        return EventCategory.HUMAN
    if name.startswith(("POLICY_", "GUARDRAIL_", "BUDGET_")):
        return EventCategory.POLICY
    if name.startswith(("ADAPTIVE_", "SCALE_", "CAPACITY_", "AUTONOMY_", "HOOK_", "RECOVERY_")):
        return EventCategory.PLATFORM
    if name in {
        "VALIDATION_STARTED",
        "VALIDATION_PASSED",
        "VALIDATION_FAILED",
        "DECISION_EMITTED",
        "HANDOFF_INITIATED",
        "HANDOFF_COMPLETED",
        "DELEGATION_GRANTED",
        "GRAPH_BACKPRESSURE",
        "TRACE_PERSISTED",
        "TASK_PROGRESS",
        "LLM_CALL",
    }:
        return EventCategory.AGENT
    if name in {"CANCELLATION_REQUESTED", "CANCELLED"}:
        return EventCategory.TASK
    if name == "RUNTIME_HANDLER_FAILED":
        return EventCategory.HUMAN
    return EventCategory.PLATFORM


def category_for_event_kind(event_kind: str) -> EventCategory:
    """Derive category from namespaced ``event_kind`` (extension path)."""
    if event_kind.startswith("agents."):
        return EventCategory.AGENT
    if event_kind.startswith("applications."):
        return EventCategory.PLATFORM
    if event_kind.startswith("platform.task."):
        return EventCategory.TASK
    if event_kind.startswith("platform.plan."):
        return EventCategory.PLAN
    if event_kind.startswith("platform.context."):
        return EventCategory.CONTEXT
    if event_kind.startswith("platform.policy."):
        return EventCategory.POLICY
    if event_kind.startswith("platform."):
        return EventCategory.PLATFORM
    if event_kind.startswith("intergrax.llm.stream."):
        return EventCategory.AGENT
    return EventCategory.PLATFORM


def consolidation_kind_for_spine_type(event_type: RuntimeEventType) -> str | None:
    """Target ``platform.*`` kind when member consolidates to ``DOMAIN_SIGNAL``."""
    name = event_type.name
    if name.startswith("ADAPTIVE_"):
        return f"platform.adaptive.{event_type.value}"
    if name.startswith(("SCALE_", "CAPACITY_", "AUTONOMY_")):
        return f"platform.capacity.{event_type.value}"
    if name.startswith("HOOK_"):
        return f"platform.hook.{event_type.value}"
    if name == "RECOVERY_REBOOT":
        return "platform.recovery.reboot"
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
    catalog: dict[RuntimeEventType, EventCatalogEntry] = {}
    for event_type in RuntimeEventType:
        phase = _SPINE_PHASE.get(event_type)
        ops_hint = _SPINE_OPS_HINT.get(event_type)
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
