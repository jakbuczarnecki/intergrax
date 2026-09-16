# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Deterministic spine metadata for canonical ``RuntimeEvent`` (OBS-CONTRACT-BOUNDARY-1-R2)."""

from __future__ import annotations

from intergrax.contracts.event_taxonomy import EventCategory
from intergrax.contracts.runtime_event_type import RuntimeEventType

OpsFilterHint = str

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
    RuntimeEventType.EXECUTION_FAILED: "ops:alert",
    RuntimeEventType.EXTERNAL_OPERATION_FAILED: "ops:alert",
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
    RuntimeEventType.LLM_CALL: "ops:llm_audit",
    RuntimeEventType.POLICY_DECISION: "ops:policy_audit",
    RuntimeEventType.GRAPH_BACKPRESSURE: "ops:backpressure",
    RuntimeEventType.GUARDRAIL_BLOCKED: "ops:alert",
    RuntimeEventType.BUDGET_THRESHOLD: "ops:budget",
    RuntimeEventType.BUDGET_EXCEEDED: "ops:alert",
    RuntimeEventType.DOMAIN_SIGNAL: "ops:domain_signal",
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
    if name == "DOMAIN_SIGNAL":
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


def spine_ops_filter_hint_for_type(
    event_type: RuntimeEventType,
) -> OpsFilterHint | None:
    return _SPINE_OPS_HINT.get(event_type)


__all__ = [
    "OpsFilterHint",
    "category_for_spine_type",
    "spine_ops_filter_hint_for_type",
]
