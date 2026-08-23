# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    reset_active_execution_identity,
    transition_active_execution_identity,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_trace import lifecycle_with_trace, TaskTraceEmitter


def _task() -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="echo.basic"),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_dual_emits_trace_and_runtime_events():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    bus = RuntimeEventBus()
    lifecycle, emitter = lifecycle_with_trace(run_id, attempt_id, event_bus=bus)
    task = _task()

    lifecycle.transition(task, TaskState.CLASSIFIED)

    assert len(emitter.events) == 1
    assert emitter.events[0].run_id == run_id
    assert len(bus.history) == 1
    assert bus.history[0].event_type == RuntimeEventType.TASK_CLASSIFIED
    assert bus.history[0].run_id == run_id
    assert bus.history[0].attempt_id == attempt_id


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_basic_emission_with_active_identity():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id, event_bus=bus)
    try:
        emitter.emit(_task(), message="probe")
        assert len(bus.history) == 1
        assert bus.history[0].run_id == run_id
        assert bus.history[0].attempt_id == attempt_id
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_follows_retry_attempt_transition():
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_a, event_bus=bus)
    try:
        emitter.emit(_task(), message="before retry")
        attempt_b = transition_active_execution_identity()
        emitter.emit(_task(), message="after retry")

        assert bus.history[0].attempt_id == attempt_a
        assert bus.history[1].attempt_id == attempt_b
        assert bus.history[0].run_id == run_id
        assert bus.history[1].run_id == run_id
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_emit_trace_step_follows_retry_attempt_transition():
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_a, event_bus=bus)
    try:
        emitter.emit_trace_step(
            _task(),
            component=TraceComponent.ENGINE,
            step="probe",
            message="before retry",
        )
        attempt_b = transition_active_execution_identity()
        emitter.emit_trace_step(
            _task(),
            component=TraceComponent.ENGINE,
            step="probe",
            message="after retry",
        )

        assert bus.history[0].attempt_id == attempt_a
        assert bus.history[1].attempt_id == attempt_b
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_multiple_retries_use_current_attempt_id():
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_a, event_bus=bus)
    try:
        emitter.emit(_task(), message="attempt a")
        attempt_b = transition_active_execution_identity()
        emitter.emit(_task(), message="attempt b")
        attempt_c = transition_active_execution_identity()
        emitter.emit(_task(), message="attempt c")

        assert [event.attempt_id for event in bus.history] == [
            attempt_a,
            attempt_b,
            attempt_c,
        ]
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_rejects_active_run_id_conflict():
    run_a = mint_run_id()
    run_b = mint_run_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_b, attempt_id=attempt_b)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_a, attempt_id=attempt_a, event_bus=bus)
    try:
        with pytest.raises(RuntimeError, match="run_id conflicts with active execution identity"):
            emitter.emit(_task(), message="conflict")
        assert bus.history == []
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_without_active_identity_uses_explicit_attempt_id():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id, event_bus=bus)
    emitter.emit(_task(), message="standalone")

    assert len(bus.history) == 1
    assert bus.history[0].run_id == run_id
    assert bus.history[0].attempt_id == attempt_id


@pytest.mark.unit
@pytest.mark.gate
def test_trace_bridge_rejects_conflicting_explicit_attempt_id_with_active_identity():
    task = _task()
    run_id = mint_run_id()
    attempt_active = mint_attempt_id()
    attempt_stale = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_active)
    trace = TraceEvent(
        event_id=TraceEvent.new_id(),
        run_id=run_id,
        seq=1,
        ts_utc="2026-06-19T10:00:00Z",
        level=TraceLevel.INFO,
        component=TraceComponent.ENGINE,
        step="diag",
        message="stale attempt",
        tags={"task_id": task.task_id},
    )
    try:
        with pytest.raises(
            RuntimeError,
            match="attempt_id conflicts with active execution identity",
        ):
            trace_event_to_runtime_event(
                trace,
                task,
                run_id=run_id,
                attempt_id=attempt_stale,
            )
    finally:
        reset_active_execution_identity(token)
