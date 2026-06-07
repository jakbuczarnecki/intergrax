# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Bridge between Nexus ``TraceEvent`` pipeline and §42 ``RuntimeEvent`` model.

Does NOT replace trace storage — publishes a canonical runtime view alongside
existing ``RunTraceWriter`` / ``TaskTraceEmitter`` (architecture §5.2, §42.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task, TaskState

TraceBridgeSubject = Union[Task, "TraceBridgeSubjectView"]


@dataclass(frozen=True)
class TraceBridgeSubjectView:
    tenant_id: str
    task_id: str
    agent_id: str = ""


def trace_bridge_subject_from_tags(
    *,
    tenant_id: str,
    task_id: str,
    agent_id: str = "",
) -> TraceBridgeSubjectView:
    return TraceBridgeSubjectView(
        tenant_id=tenant_id.strip() or "default",
        task_id=task_id.strip() or "unknown",
        agent_id=agent_id.strip(),
    )

_CORE_LLM_CALL_SCHEMA = CoreLLMCallRecordedDiagV1.schema_id()
_CORE_LLM_RETURNED_SCHEMA = "intergrax.diag.engine.core_llm.adapter_returned"

_TOOL_STEP_TO_EVENT: dict[str, RuntimeEventType] = {
    "tool_invocation_start": RuntimeEventType.TOOL_REQUESTED,
    "tool_invocation_end": RuntimeEventType.TOOL_COMPLETED,
    "tool_invocation_denied": RuntimeEventType.TOOL_DENIED,
    "tool_invocation_error": RuntimeEventType.TOOL_FAILED,
}

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


def _resolve_event_type_from_trace(
    trace: TraceEvent,
    *,
    payload_schema_id: Optional[str] = None,
    payload_dict: Optional[Dict[str, Any]] = None,
) -> tuple[RuntimeEventType, ExecutionPhase]:
    task_state_str = trace.tags.get("task_state")
    event_type = RuntimeEventType.STEP_STARTED
    phase = ExecutionPhase.STEP_EXECUTION

    schema_id = payload_schema_id or ""
    payload = dict(payload_dict or {})
    if trace.payload is not None and not payload:
        payload = trace.payload.to_dict()
        schema_id = schema_id or trace.payload.__class__.schema_id()

    if schema_id in {_CORE_LLM_CALL_SCHEMA, _CORE_LLM_RETURNED_SCHEMA}:
        event_type = RuntimeEventType.LLM_CALL
    elif trace.step == "core_llm" and "finish_reason" in payload:
        event_type = RuntimeEventType.LLM_CALL
    elif trace.step in _TOOL_STEP_TO_EVENT:
        event_type = _TOOL_STEP_TO_EVENT[trace.step]
    elif trace.message.startswith("retry attempt"):
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

    return event_type, phase


def trace_event_to_runtime_event(
    trace: TraceEvent,
    subject: TraceBridgeSubject,
    *,
    correlation_id: Optional[str] = None,
    payload_schema_id: Optional[str] = None,
    payload_dict: Optional[Dict[str, Any]] = None,
) -> RuntimeEvent:
    """Map a persisted ``TraceEvent`` to canonical ``RuntimeEvent``."""
    event_type, phase = _resolve_event_type_from_trace(
        trace,
        payload_schema_id=payload_schema_id,
        payload_dict=payload_dict,
    )

    payload: Dict[str, Any] = {
        "trace_event_id": trace.event_id,
        "trace_step": trace.step,
        "trace_component": trace.component.value,
        "trace_seq": trace.seq,
        "message": trace.message,
        "tags": dict(trace.tags),
        "source": "trace_bridge",
    }
    schema_id = payload_schema_id or ""
    extra_payload = dict(payload_dict or {})
    if trace.payload is not None and not extra_payload:
        extra_payload = trace.payload.to_dict()
        schema_id = schema_id or trace.payload.__class__.schema_id()
    if schema_id:
        payload["payload_schema_id"] = schema_id
    if extra_payload:
        payload["trace_payload"] = extra_payload
    if event_type == RuntimeEventType.LLM_CALL and extra_payload:
        payload.update(
            {
                "model": extra_payload.get("model", ""),
                "prompt_tokens": int(extra_payload.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(extra_payload.get("completion_tokens", 0) or 0),
                "total_tokens": int(extra_payload.get("total_tokens", 0) or 0),
                "finish_reason": extra_payload.get("finish_reason"),
            }
        )
    elif event_type in {
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.TOOL_DENIED,
        RuntimeEventType.TOOL_FAILED,
    }:
        tool_name = extra_payload.get("tool_name") or trace.tags.get("tool_name")
        if tool_name:
            payload["tool_name"] = tool_name

    mapped_phase = phase_for_event(event_type)
    if mapped_phase is not None:
        phase = mapped_phase

    return RuntimeEvent(
        event_id=f"rt_{trace.event_id}",
        tenant_id=str(trace.tags.get("tenant_id") or subject.tenant_id),
        task_id=str(trace.tags.get("task_id") or subject.task_id),
        run_id=trace.run_id,
        agent_id=trace.tags.get("agent_id") or subject.agent_id or None,
        event_type=event_type,
        phase=phase,
        severity=_trace_level_to_severity(trace.level),
        payload=payload,
        timestamp=_parse_timestamp(trace.ts_utc),
        correlation_id=correlation_id or subject.task_id,
        schema_version="runtime_event.v1",
    )
