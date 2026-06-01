# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.phase_coverage import phase_for_event
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
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
