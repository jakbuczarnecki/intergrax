# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.llm_adapters.registry.catalog_miss_diag import (
    CatalogResolutionTier,
    reset_catalog_miss_diagnostics,
)
from intergrax.llm_adapters.registry.context_window import resolve_context_window_tokens
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.adapters.model_catalog_miss import (
    ModelCatalogMissTraceDiagV1,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.task.task import Task


@pytest.mark.integration
@pytest.mark.gate
def test_configure_llm_tracker_wires_catalog_miss_without_core_adapter() -> None:
    reset_catalog_miss_diagnostics()
    config = RuntimeConfig(
        llm_adapter=None,
        enable_rag=False,
        enable_websearch=False,
        tools_mode="off",
        production_mode=False,
    )
    session_manager = SessionManager(storage=InMemorySessionStorage())
    ctx = RuntimeContext.build(config=config, session_manager=session_manager)
    request = RuntimeRequest(
        tenant_id="tenant-1",
        agent_id="agent-1",
        session_id="sess-1",
        user_id="user-1",
        message="hello",
        metadata={"task_id": "task-miss"},
    )
    state = RuntimeState(context=ctx, request=request, run_id="run-miss-wire")

    resolve_context_window_tokens(
        "openrouter",
        "vendor/pending-model",
        profile_options={"run_id": "run-miss-wire"},
    )
    assert len(state.trace_events) == 0

    state.configure_llm_tracker()
    assert len(state.trace_events) == 1
    trace = state.trace_events[0]
    assert trace.step == "llm_catalog_miss"
    assert isinstance(trace.payload, ModelCatalogMissTraceDiagV1)
    assert trace.payload.resolution_tier == CatalogResolutionTier.PROVIDER_DEFAULT.value


@pytest.mark.integration
@pytest.mark.gate
def test_catalog_miss_trace_maps_to_runtime_bus_llm_call() -> None:
    task = Task(
        task_id="task-miss",
        tenant_id="tenant-1",
        user_id="user-1",
        agent_id="agent-1",
        message="hello",
    )
    trace = TraceEvent(
        event_id="llm-miss-e2e",
        run_id="run-e2e",
        seq=9,
        ts_utc="2026-06-19T12:00:00Z",
        level=TraceLevel.WARNING,
        component=TraceComponent.ENGINE,
        step="llm_catalog_miss",
        message="Model catalog miss — context window resolved without exact catalog entry.",
        tags={"task_id": "task-miss"},
        payload=ModelCatalogMissTraceDiagV1(
            provider_slug="openrouter",
            model_id="vendor/unknown-e2e",
            resolved_tokens=128_000,
            resolution_tier=CatalogResolutionTier.PROVIDER_DEFAULT.value,
            run_id="run-e2e",
        ),
    )
    payload_dict = trace.payload.to_dict() if trace.payload is not None else {}
    event = trace_event_to_runtime_event(
        trace,
        task,
        payload_schema_id=ModelCatalogMissTraceDiagV1.schema_id(),
        payload_dict=payload_dict,
    )
    assert event.event_type == RuntimeEventType.LLM_CALL
    assert event.payload["model"] == "vendor/unknown-e2e"
    assert event.payload["resolution_tier"] == CatalogResolutionTier.PROVIDER_DEFAULT.value
    assert event.payload["resolved_tokens"] == 128_000
