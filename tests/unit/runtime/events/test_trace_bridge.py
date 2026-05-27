# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import (
    runtime_event_from_task_state,
    trace_event_to_runtime_event,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_event_from_task_state_completed():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        state=TaskState.COMPLETED,
        context=TaskContext(capability="echo.basic"),
    )
    event = runtime_event_from_task_state(task, run_id="run_1")
    assert event.event_type == RuntimeEventType.TASK_COMPLETED
    assert event.task_id == task.task_id
    assert event.run_id == "run_1"


@pytest.mark.unit
@pytest.mark.gate
def test_trace_event_to_runtime_event_lifecycle():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        state=TaskState.CLASSIFIED,
        context=TaskContext(capability="echo.basic"),
    )
    trace = TraceEvent(
        event_id="trace-1",
        run_id=task.task_id,
        seq=1,
        ts_utc="2026-05-27T12:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="task state -> classified",
        tags={
            "task_id": task.task_id,
            "task_state": TaskState.CLASSIFIED.value,
            "agent_id": None,
            "capability": "echo.basic",
        },
    )
    event = trace_event_to_runtime_event(trace, task)
    assert event.event_type == RuntimeEventType.TASK_CLASSIFIED
    assert event.payload["trace_event_id"] == "trace-1"


@pytest.mark.unit
@pytest.mark.gate
def test_trace_event_to_runtime_event_retry():
    task = Task(tenant_id="t1", user_id="u1", message="hi")
    trace = TraceEvent(
        event_id="trace-2",
        run_id=task.task_id,
        seq=2,
        ts_utc="2026-05-27T12:00:01+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step="task_lifecycle",
        message="retry attempt 1: validation -> echo",
        tags={"task_id": task.task_id, "task_state": TaskState.RUNNING.value},
    )
    event = trace_event_to_runtime_event(trace, task)
    assert event.event_type == RuntimeEventType.RETRY_STARTED
