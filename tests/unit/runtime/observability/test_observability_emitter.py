# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from testing_support.runtime_events import runtime_event_test_identity
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.observability import ObservabilityEmitter, TraceScope
from testing_support.builder import FakeLLMAdapter

pytestmark = pytest.mark.gate


def _runtime_state_with_bus() -> tuple[RuntimeState, InMemoryRuntimeEventStore, str, str]:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        runtime_event_bus=bus,
        enable_rag=False,
        enable_websearch=False,
        production_mode=False,
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    ctx = RuntimeContext.build(config=config, session_manager=session_manager)
    state = RuntimeState(
        context=ctx,
        request=RuntimeRequest(
            agent_id="agent-1",
            user_id="user-1",
            session_id="sess-1",
            message="hello",
            tenant_id="tenant-1",
            task_id=task_id,
            run_id=run_id,
            metadata={"task_id": task_id},
        ),
        run_id=run_id,
    )
    return state, store, task_id, run_id


def test_runtime_state_trace_event_delegates_to_emitter() -> None:
    state, store, task_id, run_id = _runtime_state_with_bus()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="core_llm",
            message="test llm trace",
            level=TraceLevel.INFO,
        )
    finally:
        reset_active_execution_identity(token)
    events = store.list_for_task(task_id, tenant_id="tenant-1")
    assert len(events) == 1
    assert events[0].event_type == RuntimeEventType.STEP_STARTED
    assert len(state.trace_events) == 1


def test_trace_scope_step_builds_causal_parent_chain() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    identity = runtime_event_test_identity()
    task_id = str(identity["task_id"])
    run_id = str(identity["run_id"])
    emitter = ObservabilityEmitter(
        run_id=run_id,
        task_id=task_id,
        tenant_id="tenant-1",
        agent_id="agent-1",
        attempt_id=str(identity["attempt_id"]),
        event_bus=bus,
    )
    token = bind_active_execution_identity(
        run_id=identity["run_id"],
        attempt_id=identity["attempt_id"],
        execution_id=identity["execution_id"],
    )
    try:
        with emitter.run() as scope:
            with scope.step("tool.invoke"):
                emitter.emit_diagnostic(
                    component=TraceComponent.TOOLS,
                    step="tool_invocation_start",
                    message="tool start",
                )
    finally:
        reset_active_execution_identity(token)

    events = store.list_for_task(task_id, tenant_id="tenant-1")
    assert len(events) == 3
    started = next(e for e in events if e.event_type == RuntimeEventType.STEP_STARTED)
    tool_evt = next(e for e in events if e.event_type == RuntimeEventType.TOOL_REQUESTED)
    completed = next(e for e in events if e.event_type == RuntimeEventType.STEP_COMPLETED)

    assert started.step_id == "tool.invoke"
    assert tool_evt.parent_event_id == started.event_id
    assert completed.parent_event_id == started.event_id


def test_emitter_preserves_captured_execution_id_after_context_reset() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    identity = runtime_event_test_identity()
    task_id = str(identity["task_id"])
    run_id = str(identity["run_id"])
    execution_id = str(identity["execution_id"])
    token = bind_active_execution_identity(
        run_id=identity["run_id"],
        attempt_id=identity["attempt_id"],
        execution_id=identity["execution_id"],
    )
    try:
        emitter = ObservabilityEmitter(
            run_id=run_id,
            task_id=task_id,
            tenant_id="tenant-1",
            agent_id="agent-1",
            attempt_id=str(identity["attempt_id"]),
            execution_id=execution_id,
            event_bus=bus,
        )
    finally:
        reset_active_execution_identity(token)

    token = bind_active_execution_identity(
        run_id=identity["run_id"],
        attempt_id=identity["attempt_id"],
    )
    try:
        emitter.emit_diagnostic(
            component=TraceComponent.ENGINE,
            step="execution_identity_probe",
            message="probe",
        )
    finally:
        reset_active_execution_identity(token)

    events = store.list_for_task(task_id, tenant_id="tenant-1")
    assert len(events) == 1
    assert str(events[0].execution_id) == execution_id


def test_nested_trace_scope_inherits_correlation_id() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    identity = runtime_event_test_identity()
    task_id = str(identity["task_id"])
    run_id = str(identity["run_id"])
    emitter = ObservabilityEmitter(
        run_id=run_id,
        task_id=task_id,
        tenant_id="tenant-1",
        attempt_id=str(identity["attempt_id"]),
        event_bus=bus,
    )
    token = bind_active_execution_identity(
        run_id=identity["run_id"],
        attempt_id=identity["attempt_id"],
        execution_id=identity["execution_id"],
    )
    try:
        parent = emitter.emit_runtime(
            RuntimeEvent(
                event_type=RuntimeEventType.TASK_CREATED,
                phase=ExecutionPhase.INTAKE,
                correlation_id="corr-root",
                **identity,
            )
        )
        with TraceScope(
            emitter,
            run_id=run_id,
            task_id=task_id,
            tenant_id="tenant-1",
            correlation_id="corr-root",
            parent_event_id=parent.event_id,
        ):
            child = emitter.emit_diagnostic(
                component=TraceComponent.ENGINE,
                step="core_llm",
                message="nested",
            )
    finally:
        reset_active_execution_identity(token)

    assert child.runtime.parent_event_id == parent.event_id
    assert child.runtime.correlation_id == "corr-root"
