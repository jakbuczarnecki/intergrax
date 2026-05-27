# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_trace import lifecycle_with_trace


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_dual_emits_trace_and_runtime_events():
    bus = RuntimeEventBus()
    lifecycle, emitter = lifecycle_with_trace("run_gate_1", event_bus=bus)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="echo.basic"),
    )

    lifecycle.transition(task, TaskState.CLASSIFIED)

    assert len(emitter.events) == 1
    assert emitter.events[0].run_id == "run_gate_1"
    assert len(bus.history) == 1
    assert bus.history[0].event_type == RuntimeEventType.TASK_CLASSIFIED
    assert bus.history[0].run_id == "run_gate_1"
