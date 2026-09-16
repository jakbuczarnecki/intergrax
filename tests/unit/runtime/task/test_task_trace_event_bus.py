# © Artur Czarnecki. All rights reserved.

from contextvars import Token

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
    transition_active_execution_identity,
)
from intergrax.runtime.task.task import Task
from intergrax.dev_support.execution_identity_scope import refresh_active_execution_id_for_tests
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import TaskState
from intergrax.runtime.task.task_trace import lifecycle_with_trace, TaskTraceEmitter
from testing_support.builder import build_task_for_tests, canonical_run_id_for_tests


def _task(seed: str = "task-trace") -> Task:
    return build_task_for_tests(seed=seed, tenant_id="t1", user_id="u1", message="hi")


def _bind_trace_identity(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> Token:
    return bind_active_execution_identity(
        run_id=run_id if run_id is not None else mint_run_id(),
        attempt_id=attempt_id if attempt_id is not None else mint_attempt_id(),
        execution_id=execution_id if execution_id is not None else mint_execution_id(),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_task_trace_emitter_dual_emits_trace_and_runtime_events():
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    bus = RuntimeEventBus()
    lifecycle, emitter = lifecycle_with_trace(run_id, attempt_id, event_bus=bus)
    task = _task()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        lifecycle.transition(task, TaskState.CLASSIFIED)
    finally:
        reset_active_execution_identity(token)

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
    token = _bind_trace_identity(run_id=run_id, attempt_id=attempt_id)
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
    token = _bind_trace_identity(run_id=run_id, attempt_id=attempt_a)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_a, event_bus=bus)
    try:
        emitter.emit(_task(), message="before retry")
        attempt_b = transition_active_execution_identity()
        refresh_active_execution_id_for_tests()
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
    token = _bind_trace_identity(run_id=run_id, attempt_id=attempt_a)
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
        refresh_active_execution_id_for_tests()
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
    token = _bind_trace_identity(run_id=run_id, attempt_id=attempt_a)
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_a, event_bus=bus)
    try:
        emitter.emit(_task(), message="attempt a")
        attempt_b = transition_active_execution_identity()
        refresh_active_execution_id_for_tests()
        emitter.emit(_task(), message="attempt b")
        attempt_c = transition_active_execution_identity()
        refresh_active_execution_id_for_tests()
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
    token = bind_active_execution_identity(
        run_id=run_b,
        attempt_id=attempt_b,
        execution_id=mint_execution_id(),
    )
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
    seed = "trace-standalone"
    run_id = canonical_run_id_for_tests(seed)
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    bus = RuntimeEventBus()
    emitter = TaskTraceEmitter(run_id=run_id, attempt_id=attempt_id, event_bus=bus)
    try:
        emitter.emit(_task(seed), message="standalone")

        assert len(bus.history) == 1
        assert bus.history[0].run_id == run_id
        assert bus.history[0].attempt_id == attempt_id
    finally:
        reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
def test_trace_bridge_rejects_conflicting_explicit_attempt_id_with_active_identity():
    task = _task()
    run_id = mint_run_id()
    attempt_active = mint_attempt_id()
    attempt_stale = mint_attempt_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_active,
        execution_id=mint_execution_id(),
    )
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
