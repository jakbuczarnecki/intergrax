# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task, TaskState

pytestmark = pytest.mark.gate


def test_phase_for_event_matches_trace_bridge_retry() -> None:
    task = Task(
        task_id="t1",
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    )
    trace = TraceEvent(
        event_id="e1",
        run_id="r1",
        seq=1,
        ts_utc="2026-06-01T00:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="retry",
        message="retry attempt 1",
        tags={"task_id": "t1", "task_state": TaskState.RUNNING.value},
    )
    event = trace_event_to_runtime_event(trace, task)
    assert event.event_type == RuntimeEventType.RETRY_STARTED
    assert event.phase == phase_for_event(RuntimeEventType.RETRY_STARTED)
    assert event.phase == ExecutionPhase.RETRY_HANDLING


def test_trace_bridge_maps_tool_invocation_start() -> None:
    task = Task(
        task_id="t1",
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    )
    trace = TraceEvent(
        event_id="tool-1",
        run_id="r1",
        seq=3,
        ts_utc="2026-06-07T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.TOOLS,
        step="tool_invocation_start",
        message="invoke rag.retrieve",
        tags={"task_id": "t1", "tool_name": "rag.retrieve"},
    )
    event = trace_event_to_runtime_event(
        trace,
        task,
        payload_dict={"tool_name": "rag.retrieve"},
    )
    assert event.event_type == RuntimeEventType.TOOL_REQUESTED
    assert event.payload["tool_name"] == "rag.retrieve"
    assert event.payload["trace_seq"] == 3


def test_trace_bridge_maps_llm_call_recorded_schema() -> None:
    task = Task(
        task_id="t1",
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    )
    trace = TraceEvent(
        event_id="llm-1",
        run_id="r1",
        seq=4,
        ts_utc="2026-06-07T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="core_llm",
        message="call recorded",
        tags={"task_id": "t1"},
    )
    payload = {
        "model": "gpt-test",
        "provider": "openai",
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
        "finish_reason": "stop",
        "response_id": "resp-1",
        "has_refusal": False,
        "has_tool_calls": False,
    }
    event = trace_event_to_runtime_event(
        trace,
        task,
        payload_schema_id=CoreLLMCallRecordedDiagV1.schema_id(),
        payload_dict=payload,
    )
    assert event.event_type == RuntimeEventType.LLM_CALL
    assert event.payload["model"] == "gpt-test"
    assert event.payload["total_tokens"] == 16
