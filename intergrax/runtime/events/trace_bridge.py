# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Bridge between Nexus ``TraceEvent`` pipeline and §42 ``RuntimeEvent`` model.

Does NOT replace trace storage — publishes a canonical runtime view alongside
existing ``RunTraceWriter`` / ``TaskTraceEmitter`` (architecture §5.2, §42.1).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task, TaskState

_TASK_STATE_TO_EVENT: dict[TaskState, RuntimeEventType] = {
    TaskState.CREATED: RuntimeEventType.TASK_CREATED,
    TaskState.CLASSIFIED: RuntimeEventType.TASK_CLASSIFIED,
    TaskState.PLANNED: RuntimeEventType.PLAN_CREATED,
    TaskState.WAITING_FOR_RESOURCES: RuntimeEventType.PAUSE_REQUESTED,
    TaskState.WAITING_FOR_HUMAN: RuntimeEventType.HUMAN_APPROVAL_REQUESTED,
    TaskState.RUNNING: RuntimeEventType.STEP_STARTED,
    TaskState.VALIDATING: RuntimeEventType.VALIDATION_STARTED,
    TaskState.COMPLETED: RuntimeEventType.TASK_COMPLETED,
    TaskState.PARTIALLY_COMPLETED: RuntimeEventType.TASK_COMPLETED,
    TaskState.NEEDS_MORE_INFORMATION: RuntimeEventType.TASK_FAILED,
    TaskState.FAILED: RuntimeEventType.TASK_FAILED,
    TaskState.CANCELLED: RuntimeEventType.CANCELLED,
    TaskState.EXPIRED: RuntimeEventType.TASK_FAILED,
}

_TASK_STATE_TO_PHASE: dict[TaskState, ExecutionPhase] = {
    TaskState.CREATED: ExecutionPhase.INTAKE,
    TaskState.CLASSIFIED: ExecutionPhase.CLASSIFICATION,
    TaskState.PLANNED: ExecutionPhase.PLANNING,
    TaskState.WAITING_FOR_HUMAN: ExecutionPhase.HUMAN_APPROVAL,
    TaskState.RUNNING: ExecutionPhase.STEP_EXECUTION,
    TaskState.VALIDATING: ExecutionPhase.VALIDATION,
    TaskState.COMPLETED: ExecutionPhase.COMPLETION,
    TaskState.PARTIALLY_COMPLETED: ExecutionPhase.COMPLETION,
    TaskState.FAILED: ExecutionPhase.COMPLETION,
    TaskState.CANCELLED: ExecutionPhase.COMPLETION,
}


def _trace_level_to_severity(level: TraceLevel) -> EventSeverity:
    if level == TraceLevel.ERROR:
        return EventSeverity.ERROR
    if level == TraceLevel.WARNING:
        return EventSeverity.WARNING
    if level == TraceLevel.DEBUG:
        return EventSeverity.DEBUG
    return EventSeverity.INFO


def _parse_timestamp(ts_utc: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_utc.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def runtime_event_from_task_state(
    task: Task,
    *,
    run_id: str,
    message: str = "",
    correlation_id: Optional[str] = None,
) -> RuntimeEvent:
    event_type = _TASK_STATE_TO_EVENT.get(task.state, RuntimeEventType.STEP_STARTED)
    phase = _TASK_STATE_TO_PHASE.get(task.state, ExecutionPhase.STEP_EXECUTION)
    return RuntimeEvent(
        tenant_id=task.tenant_id,
        task_id=task.task_id,
        run_id=run_id,
        agent_id=task.agent_id,
        event_type=event_type,
        phase=phase,
        severity=EventSeverity.INFO,
        payload={
            "task_state": task.state.value,
            "message": message,
            "capability": task.context.capability,
            "source": "task_lifecycle",
        },
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or task.task_id,
    )


def trace_event_to_runtime_event(
    trace: TraceEvent,
    task: Task,
    *,
    correlation_id: Optional[str] = None,
) -> RuntimeEvent:
    """Map a persisted ``TraceEvent`` to canonical ``RuntimeEvent``."""
    task_state_str = trace.tags.get("task_state")
    event_type = RuntimeEventType.STEP_STARTED
    phase = ExecutionPhase.STEP_EXECUTION

    if trace.message.startswith("retry attempt"):
        event_type = RuntimeEventType.RETRY_STARTED
        phase = ExecutionPhase.RETRY_HANDLING
    elif trace.step == "task_lifecycle" and task_state_str:
        try:
            state = TaskState(task_state_str)
            event_type = _TASK_STATE_TO_EVENT.get(state, RuntimeEventType.STEP_STARTED)
            phase = _TASK_STATE_TO_PHASE.get(state, ExecutionPhase.STEP_EXECUTION)
        except ValueError:
            pass
    elif trace.message.startswith("graph node start:"):
        event_type = RuntimeEventType.STEP_STARTED
        phase = ExecutionPhase.STEP_EXECUTION
    elif trace.message.startswith("graph node complete:"):
        event_type = RuntimeEventType.STEP_COMPLETED
        phase = ExecutionPhase.STEP_EXECUTION

    payload: Dict[str, Any] = {
        "trace_event_id": trace.event_id,
        "trace_step": trace.step,
        "trace_component": trace.component.value,
        "message": trace.message,
        "tags": dict(trace.tags),
    }

    return RuntimeEvent(
        event_id=f"rt_{trace.event_id}",
        tenant_id=str(trace.tags.get("tenant_id") or task.tenant_id),
        task_id=str(trace.tags.get("task_id") or task.task_id),
        run_id=trace.run_id,
        agent_id=trace.tags.get("agent_id") or task.agent_id,
        event_type=event_type,
        phase=phase,
        severity=_trace_level_to_severity(trace.level),
        payload=payload,
        timestamp=_parse_timestamp(trace.ts_utc),
        correlation_id=correlation_id or task.task_id,
        schema_version="runtime_event.v1",
    )
