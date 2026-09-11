# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit RuntimeEvent → observability delivery classification (W5-B)."""

from __future__ import annotations

from intergrax.contracts.event_delivery import (
    CriticalEventKind,
    DeliverableEvent,
    EventPriority,
    classify_kind_string,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


class UnclassifiedRuntimeEventDeliveryError(ValueError):
    """Raised when a runtime event has no explicit delivery priority mapping."""


_RUNTIME_EVENT_DELIVERY_PRIORITY: dict[RuntimeEventType, EventPriority] = {
    RuntimeEventType.TASK_CREATED: EventPriority.BEST_EFFORT,
    RuntimeEventType.TASK_CLASSIFIED: EventPriority.BEST_EFFORT,
    RuntimeEventType.PLAN_CREATED: EventPriority.BEST_EFFORT,
    RuntimeEventType.PLAN_UPDATED: EventPriority.BEST_EFFORT,
    RuntimeEventType.PLAN_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.AGENT_SELECTED: EventPriority.BEST_EFFORT,
    RuntimeEventType.CONTEXT_BUILT: EventPriority.BEST_EFFORT,
    RuntimeEventType.CONTEXT_ASSEMBLED: EventPriority.BEST_EFFORT,
    RuntimeEventType.CONTEXT_TRIMMED: EventPriority.BEST_EFFORT,
    RuntimeEventType.CONTEXT_CANDIDATE_COLLECTED: EventPriority.BEST_EFFORT,
    RuntimeEventType.CONTEXT_CANDIDATE_DROPPED: EventPriority.BEST_EFFORT,
    RuntimeEventType.CONTEXT_VALIDATION_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.INGESTION_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.SKILL_RESOLVED: EventPriority.BEST_EFFORT,
    RuntimeEventType.SKILL_IMPORT_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.STEP_STARTED: EventPriority.IMPORTANT,
    RuntimeEventType.STEP_COMPLETED: EventPriority.IMPORTANT,
    RuntimeEventType.STEP_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.EXECUTION_FAILED: EventPriority.CRITICAL,
    RuntimeEventType.TOOL_REQUESTED: EventPriority.IMPORTANT,
    RuntimeEventType.TOOL_COMPLETED: EventPriority.IMPORTANT,
    RuntimeEventType.TOOL_DENIED: EventPriority.IMPORTANT,
    RuntimeEventType.TOOL_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.VALIDATION_STARTED: EventPriority.IMPORTANT,
    RuntimeEventType.VALIDATION_PASSED: EventPriority.IMPORTANT,
    RuntimeEventType.VALIDATION_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.DECISION_EMITTED: EventPriority.CRITICAL,
    RuntimeEventType.INTERRUPT_REQUESTED: EventPriority.IMPORTANT,
    RuntimeEventType.INTERRUPT_HANDLED: EventPriority.IMPORTANT,
    RuntimeEventType.INTERRUPT_ESCALATED: EventPriority.CRITICAL,
    RuntimeEventType.HUMAN_APPROVAL_REQUESTED: EventPriority.IMPORTANT,
    RuntimeEventType.HUMAN_APPROVAL_RECEIVED: EventPriority.IMPORTANT,
    RuntimeEventType.HUMAN_APPROVAL_TIMEOUT: EventPriority.IMPORTANT,
    RuntimeEventType.PAUSE_REQUESTED: EventPriority.IMPORTANT,
    RuntimeEventType.PAUSED: EventPriority.IMPORTANT,
    RuntimeEventType.RESUMED: EventPriority.IMPORTANT,
    RuntimeEventType.RETRY_SCHEDULED: EventPriority.IMPORTANT,
    RuntimeEventType.RETRY_STARTED: EventPriority.IMPORTANT,
    RuntimeEventType.CANCELLATION_REQUESTED: EventPriority.IMPORTANT,
    RuntimeEventType.CANCELLED: EventPriority.CRITICAL,
    RuntimeEventType.MEMORY_READ: EventPriority.BEST_EFFORT,
    RuntimeEventType.MEMORY_WRITE: EventPriority.BEST_EFFORT,
    RuntimeEventType.HANDOFF_INITIATED: EventPriority.IMPORTANT,
    RuntimeEventType.HANDOFF_COMPLETED: EventPriority.IMPORTANT,
    RuntimeEventType.DELEGATION_GRANTED: EventPriority.IMPORTANT,
    RuntimeEventType.TRACE_PERSISTED: EventPriority.IMPORTANT,
    RuntimeEventType.TASK_PROGRESS: EventPriority.IMPORTANT,
    RuntimeEventType.TASK_COMPLETED: EventPriority.CRITICAL,
    RuntimeEventType.TASK_FAILED: EventPriority.CRITICAL,
    RuntimeEventType.RUNTIME_HANDLER_FAILED: EventPriority.IMPORTANT,
    RuntimeEventType.LLM_CALL: EventPriority.IMPORTANT,
    RuntimeEventType.POLICY_DECISION: EventPriority.CRITICAL,
    RuntimeEventType.GRAPH_BACKPRESSURE: EventPriority.IMPORTANT,
    RuntimeEventType.GUARDRAIL_BLOCKED: EventPriority.IMPORTANT,
    RuntimeEventType.BUDGET_THRESHOLD: EventPriority.IMPORTANT,
    RuntimeEventType.BUDGET_EXCEEDED: EventPriority.CRITICAL,
    RuntimeEventType.DOMAIN_SIGNAL: EventPriority.BEST_EFFORT,
}

_missing_types = set(RuntimeEventType) - set(_RUNTIME_EVENT_DELIVERY_PRIORITY)
if _missing_types:
    raise RuntimeError(
        "incomplete runtime event delivery priority map: "
        f"{sorted(t.value for t in _missing_types)}",
    )


def runtime_event_to_deliverable(event: RuntimeEvent) -> DeliverableEvent:
    kind = event.event_kind or event.event_type.value
    return DeliverableEvent(
        event_id=str(event.event_id),
        kind=kind,
    )


def delivery_priority_for_runtime_event(event: RuntimeEvent) -> EventPriority:
    kind = (event.event_kind or "").strip()
    if kind:
        try:
            CriticalEventKind(kind)
        except ValueError:
            if kind.startswith("metrics.") or kind.startswith("timing."):
                return EventPriority.IMPORTANT
            classified = classify_kind_string(kind)
            if classified is EventPriority.CRITICAL:
                return EventPriority.CRITICAL
        else:
            return EventPriority.CRITICAL
    mapped = _RUNTIME_EVENT_DELIVERY_PRIORITY.get(event.event_type)
    if mapped is None:
        raise UnclassifiedRuntimeEventDeliveryError(
            f"no delivery priority for runtime event type {event.event_type.value}",
        )
    return mapped
