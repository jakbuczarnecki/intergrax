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
from intergrax.runtime.events.payload_registry import merge_payload_envelope
from intergrax.runtime.events.payloads import (
    GraphNodePayloadV1,
    LlmCallPayloadV1,
    TaskLifecyclePayloadV1,
    ToolPayloadV1,
    TraceBridgePayloadV1,
    ValidationPayloadV1,
)
from intergrax.runtime.nexus.tracing.graph_node_diag import (
    GRAPH_NODE_STEP_COMPLETE,
    GRAPH_NODE_STEP_START,
    GraphNodeDiagV1,
)
from intergrax.runtime.nexus.tracing.steps.step_failed import RuntimeStepFailedDiagV1
from intergrax.runtime.nexus.tracing.steps.step_finished import RuntimeStepFinishedDiagV1
from intergrax.runtime.nexus.tracing.steps.step_started import RuntimeStepStartedDiagV1
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

_CRITIC_STEP_EVALUATOR_LOOP = "critic.evaluator_loop"

_CRITIC_STEP_TO_EVENT: dict[str, RuntimeEventType] = {
    "critic.l0_failed": RuntimeEventType.VALIDATION_FAILED,
    "critic.l1_judge": RuntimeEventType.LLM_CALL,
    "critic.trajectory": RuntimeEventType.STEP_COMPLETED,
    _CRITIC_STEP_EVALUATOR_LOOP: RuntimeEventType.STEP_COMPLETED,
    "critic.final_verdict": RuntimeEventType.VALIDATION_STARTED,
}

_RUNTIME_STEP_SCHEMA_TO_EVENT: dict[str, RuntimeEventType] = {
    RuntimeStepStartedDiagV1.schema_id(): RuntimeEventType.STEP_STARTED,
    RuntimeStepFinishedDiagV1.schema_id(): RuntimeEventType.STEP_COMPLETED,
    RuntimeStepFailedDiagV1.schema_id(): RuntimeEventType.STEP_FAILED,
}

_GRAPH_STEP_TO_EVENT: dict[str, RuntimeEventType] = {
    GRAPH_NODE_STEP_START: RuntimeEventType.STEP_STARTED,
    GRAPH_NODE_STEP_COMPLETE: RuntimeEventType.STEP_COMPLETED,
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
    from intergrax.runtime.events.payload_registry import runtime_event_with_payload

    event_type = _TASK_STATE_TO_EVENT.get(task.state, RuntimeEventType.STEP_STARTED)
    phase = _TASK_STATE_TO_PHASE.get(task.state, ExecutionPhase.STEP_EXECUTION)
    base = RuntimeEvent(
        tenant_id=task.tenant_id,
        task_id=task.task_id,
        run_id=run_id,
        agent_id=task.agent_id,
        event_type=event_type,
        phase=phase,
        severity=EventSeverity.INFO,
        timestamp=datetime.now(timezone.utc),
        correlation_id=correlation_id or task.task_id,
    )
    capability = task.context.capability or ""
    lifecycle = TaskLifecyclePayloadV1(
        task_state=task.state.value,
        message=message,
        capability=capability,
        source="task_lifecycle",
    )
    return runtime_event_with_payload(
        base,
        lifecycle,
        promote_fields={
            "task_state": task.state.value,
            "message": message,
            "capability": capability,
            "source": "task_lifecycle",
        },
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

    if schema_id in _RUNTIME_STEP_SCHEMA_TO_EVENT:
        event_type = _RUNTIME_STEP_SCHEMA_TO_EVENT[schema_id]
    elif trace.step in _GRAPH_STEP_TO_EVENT:
        event_type = _GRAPH_STEP_TO_EVENT[trace.step]
    elif schema_id == GraphNodeDiagV1.schema_id() and trace.step in _GRAPH_STEP_TO_EVENT:
        event_type = _GRAPH_STEP_TO_EVENT[trace.step]
    elif schema_id in {_CORE_LLM_CALL_SCHEMA, _CORE_LLM_RETURNED_SCHEMA}:
        event_type = RuntimeEventType.LLM_CALL
    elif trace.step == "core_llm" and "finish_reason" in payload:
        event_type = RuntimeEventType.LLM_CALL
    elif trace.step in _TOOL_STEP_TO_EVENT:
        event_type = _TOOL_STEP_TO_EVENT[trace.step]
    elif trace.step in _CRITIC_STEP_TO_EVENT:
        event_type = _CRITIC_STEP_TO_EVENT[trace.step]
        phase = ExecutionPhase.VALIDATION
        if trace.step in {_CRITIC_STEP_EVALUATOR_LOOP, "critic.final_verdict"}:
            payload_passed = payload.get("passed")
            if payload_passed is False:
                event_type = RuntimeEventType.VALIDATION_FAILED
            elif payload_passed is True:
                event_type = RuntimeEventType.STEP_COMPLETED
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


_TOOL_STATUS_BY_EVENT: dict[RuntimeEventType, str] = {
    RuntimeEventType.TOOL_REQUESTED: "requested",
    RuntimeEventType.TOOL_COMPLETED: "completed",
    RuntimeEventType.TOOL_DENIED: "denied",
    RuntimeEventType.TOOL_FAILED: "failed",
}


def _attach_typed_bridge_payload(
    *,
    event_type: RuntimeEventType,
    base: dict[str, Any],
    trace: TraceEvent,
    extra_payload: dict[str, Any],
    diagnostic_schema_id: str,
) -> dict[str, Any]:
    if event_type in {
        RuntimeEventType.STEP_STARTED,
        RuntimeEventType.STEP_COMPLETED,
    } and (
        trace.step in _GRAPH_STEP_TO_EVENT
        or diagnostic_schema_id == GraphNodeDiagV1.schema_id()
        or trace.message.startswith("graph node ")
    ):
        node_id = str(
            extra_payload.get("node_id")
            or trace.tags.get("node_id")
            or ""
        )
        status = str(extra_payload.get("status") or "")
        agent_id = str(
            extra_payload.get("agent_id") or trace.tags.get("agent_id") or ""
        )
        typed = GraphNodePayloadV1(
            node_id=node_id,
            status=status,
            agent_id=agent_id,
            message=trace.message,
        )
        return merge_payload_envelope(
            base,
            typed,
            promote_fields={"node_id": node_id} if node_id else None,
        )
    if event_type == RuntimeEventType.STEP_FAILED and extra_payload:
        step_name = str(extra_payload.get("step_name") or trace.step)
        error_type = str(extra_payload.get("error_type") or "error")
        typed = ValidationPayloadV1(
            valid=False,
            error_count=1,
            stage=step_name,
            rule_ids_failed=(error_type,),
        )
        return merge_payload_envelope(
            base,
            typed,
            promote_fields={"stage": step_name, "error_type": error_type},
        )
    if event_type == RuntimeEventType.LLM_CALL and extra_payload:
        typed = LlmCallPayloadV1(
            model=str(extra_payload.get("model", "")),
            prompt_tokens=int(extra_payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(extra_payload.get("completion_tokens", 0) or 0),
            total_tokens=int(extra_payload.get("total_tokens", 0) or 0),
            finish_reason=extra_payload.get("finish_reason"),
            label=str(extra_payload.get("label", "")),
        )
        return merge_payload_envelope(
            base,
            typed,
            promote_fields={
                "model": typed.model,
                "prompt_tokens": typed.prompt_tokens,
                "completion_tokens": typed.completion_tokens,
                "total_tokens": typed.total_tokens,
                "finish_reason": typed.finish_reason,
            },
        )
    if event_type in _TOOL_STATUS_BY_EVENT:
        tool_name = str(
            extra_payload.get("tool_id")
            or extra_payload.get("tool_name")
            or trace.tags.get("tool_name")
            or ""
        )
        typed = ToolPayloadV1(
            tool_name=tool_name,
            status=_TOOL_STATUS_BY_EVENT[event_type],
            duration_ms=int(extra_payload.get("duration_ms", 0) or 0),
            redacted_input_summary=str(extra_payload.get("redacted_input_summary", "")),
            step_id=str(extra_payload.get("step_id", "")),
        )
        promote: dict[str, Any] = {"tool_name": tool_name} if tool_name else {}
        return merge_payload_envelope(base, typed, promote_fields=promote or None)
    if trace.step == "task_lifecycle":
        task_state = str(trace.tags.get("task_state") or extra_payload.get("task_state") or "")
        typed = TaskLifecyclePayloadV1(
            task_state=task_state,
            message=trace.message,
            capability=str(trace.tags.get("capability") or ""),
            source="task_lifecycle",
        )
        return merge_payload_envelope(
            base,
            typed,
            promote_fields={
                "task_state": task_state,
                "message": trace.message,
                "capability": trace.tags.get("capability"),
                "source": "task_lifecycle",
            },
        )
    typed = TraceBridgePayloadV1(
        trace_event_id=trace.event_id,
        trace_step=trace.step,
        trace_component=trace.component.value,
        trace_seq=trace.seq,
        message=trace.message,
        diagnostic_schema_id=diagnostic_schema_id,
        diagnostic_data=dict(extra_payload),
    )
    merged = merge_payload_envelope(base, typed)
    if diagnostic_schema_id:
        merged["diagnostic_schema_id"] = diagnostic_schema_id
    if extra_payload:
        merged["trace_payload"] = dict(extra_payload)
    return merged


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
    payload = _attach_typed_bridge_payload(
        event_type=event_type,
        base=payload,
        trace=trace,
        extra_payload=extra_payload,
        diagnostic_schema_id=schema_id,
    )

    mapped_phase = phase_for_event(event_type)
    if mapped_phase is not None:
        phase = mapped_phase

    node_id = trace.tags.get("node_id")
    step_id = trace.tags.get("step_id")
    if extra_payload.get("step_name") and event_type in {
        RuntimeEventType.STEP_STARTED,
        RuntimeEventType.STEP_COMPLETED,
        RuntimeEventType.STEP_FAILED,
    }:
        step_id = step_id or extra_payload.get("step_name")

    return RuntimeEvent(
        event_id=f"rt_{trace.event_id}",
        tenant_id=str(trace.tags.get("tenant_id") or subject.tenant_id),
        task_id=str(trace.tags.get("task_id") or subject.task_id),
        run_id=trace.run_id,
        agent_id=trace.tags.get("agent_id") or subject.agent_id or None,
        node_id=str(node_id) if node_id else None,
        step_id=str(step_id) if step_id else None,
        event_type=event_type,
        phase=phase,
        severity=_trace_level_to_severity(trace.level),
        payload=payload,
        timestamp=_parse_timestamp(trace.ts_utc),
        correlation_id=correlation_id or subject.task_id,
        schema_version="runtime_event.v1",
    )
