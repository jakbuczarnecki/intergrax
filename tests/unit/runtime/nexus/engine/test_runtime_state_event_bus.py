# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from testing_support.builder import (
    FakeLLMAdapter,
    build_runtime_request_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
    canonical_task_id_for_tests,
)

pytestmark = pytest.mark.gate


def test_runtime_state_trace_event_records_on_event_bus() -> None:
    seed = "runtime-state-event-bus"
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(),
        runtime_event_bus=bus,
        enable_rag=False,
        enable_websearch=False,
        production_mode=False,
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    ctx = RuntimeContext.build(config=config, session_manager=session_manager)
    request = build_runtime_request_for_tests(
        seed=seed,
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        tenant_id="tenant-1",
    )
    state = RuntimeState(
        context=ctx,
        request=request,
        run_id=canonical_run_id_for_tests(seed),
    )
    with canonical_execution_identity_scope(seed):
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="core_llm",
            message="test llm trace",
            level=TraceLevel.INFO,
        )
    events = store.list_for_task(str(canonical_task_id_for_tests(seed)), tenant_id="tenant-1")
    assert len(events) == 1
    assert events[0].event_type == RuntimeEventType.STEP_STARTED
