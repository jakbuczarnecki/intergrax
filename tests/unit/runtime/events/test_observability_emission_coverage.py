# © Artur Czarnecki. All rights reserved.

"""OBS-BUS-3 emission coverage gate — key catalog events have production emitters."""

from __future__ import annotations

import inspect

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.trace_bridge import (
    _CRITIC_STEP_EVALUATOR_LOOP,
    _CRITIC_STEP_TO_EVENT,
    _GRAPH_STEP_TO_EVENT,
    _RUNTIME_STEP_SCHEMA_TO_EVENT,
    trace_event_to_runtime_event,
)
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.tracing.adapters.model_catalog_miss import (
    ModelCatalogMissTraceDiagV1,
)
from intergrax.runtime.nexus.tracing.graph_node_diag import GRAPH_NODE_STEP_START, GraphNodeDiagV1
from intergrax.runtime.nexus.tracing.steps.step_failed import RuntimeStepFailedDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceEvent, TraceLevel
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = pytest.mark.gate


class _StubAgent:
    def __init__(self, contract: AgentContract) -> None:
        self._contract = contract

    def get_contract(self) -> AgentContract:
        return self._contract

    def can_handle(self, task_context: TaskContext) -> object:
        from intergrax.contracts.capability import CapabilityMatchResult

        if task_context.capability == "demo.basic":
            return CapabilityMatchResult(matched=True, score=0.9, reason="stub")
        return CapabilityMatchResult(matched=False, score=0.0, reason="no match")


def _contract(**updates: object) -> AgentContract:
    return AgentContract(
        id="demo",
        name="Demo",
        description="demo",
        capabilities=["demo.basic"],
        max_steps=1,
    ).model_copy(update=updates)


def test_agent_router_emits_agent_selected() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    registry = AgentRegistry()
    registry.register(_StubAgent(_contract(id="active")))
    router = AgentRouter(registry, event_bus=bus)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="demo.basic"),
    )
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        router.route(task, run_id=run_id)
    finally:
        reset_active_execution_identity(token)
    events = store.list_for_task(task.task_id, tenant_id="t1")
    assert len(events) == 1
    assert events[0].event_type == RuntimeEventType.AGENT_SELECTED
    assert events[0].payload["payload_schema_id"] == "agent_selection.v1"
    assert events[0].payload["data"]["selected_agent_id"] == "active"
    assert events[0].run_id == run_id
    assert events[0].attempt_id == attempt_id


def test_trace_bridge_maps_runtime_step_failed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    trace = TraceEvent(
        event_id="te-1",
        run_id=run_id,
        seq=1,
        ts_utc="2026-06-08T12:00:00+00:00",
        level=TraceLevel.ERROR,
        component=TraceComponent.PIPELINE,
        step="HistoryStep",
        message="Step failed",
        payload=RuntimeStepFailedDiagV1(
            step_name="HistoryStep",
            error_type="ValueError",
            error_message="boom",
            error_repr="ValueError('boom')",
        ),
        tags={"tenant_id": "t1", "task_id": task_id, "agent_id": "a1"},
    )
    task = Task(task_id=task_id, tenant_id="t1", user_id="u1", message="x")
    runtime = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert runtime.event_type == RuntimeEventType.STEP_FAILED
    assert runtime.payload["payload_schema_id"] == "validation.v1"
    assert runtime.step_id == "HistoryStep"


def test_trace_bridge_maps_graph_node_typed_payload() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    trace = TraceEvent(
        event_id="te-2",
        run_id=run_id,
        seq=2,
        ts_utc="2026-06-08T12:00:00+00:00",
        level=TraceLevel.INFO,
        component=TraceComponent.PLANNER,
        step=GRAPH_NODE_STEP_START,
        message="graph node start: n1",
        payload=GraphNodeDiagV1(
            node_id="n1",
            status="running",
            agent_id="a1",
            capability="cap.a",
        ),
        tags={"tenant_id": "t1", "task_id": task_id, "node_id": "n1"},
    )
    task = Task(task_id=task_id, tenant_id="t1", user_id="u1", message="x")
    runtime = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert runtime.event_type == RuntimeEventType.STEP_STARTED
    assert runtime.payload["payload_schema_id"] == "graph_node.v1"
    assert runtime.node_id == "n1"


def test_trace_bridge_catalog_includes_evaluator_loop() -> None:
    assert _CRITIC_STEP_EVALUATOR_LOOP in _CRITIC_STEP_TO_EVENT
    assert RuntimeStepFailedDiagV1.schema_id() in _RUNTIME_STEP_SCHEMA_TO_EVENT
    assert GRAPH_NODE_STEP_START in _GRAPH_STEP_TO_EVENT


def test_trace_bridge_maps_llm_catalog_miss_schema() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    task = Task(task_id=task_id, tenant_id="t1", user_id="u1", message="x")
    trace = TraceEvent(
        event_id="llm-miss-cov",
        run_id=run_id,
        seq=8,
        ts_utc="2026-06-19T12:00:00+00:00",
        level=TraceLevel.WARNING,
        component=TraceComponent.ENGINE,
        step="llm_catalog_miss",
        message="Model catalog miss — context window resolved without exact catalog entry.",
        tags={"tenant_id": "t1", "task_id": task_id},
        payload=ModelCatalogMissTraceDiagV1(
            provider_slug="openrouter",
            model_id="vendor/obs-cov",
            resolved_tokens=128_000,
            resolution_tier="provider_default",
            run_id=run_id,
        ),
    )
    runtime = trace_event_to_runtime_event(
        trace,
        task,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    assert runtime.event_type == RuntimeEventType.LLM_CALL
    assert runtime.payload["model"] == "vendor/obs-cov"
    assert runtime.payload["resolution_tier"] == "provider_default"


def test_agent_router_source_records_agent_selected_constant() -> None:
    source = inspect.getsource(AgentRouter._emit_agent_selected)
    assert "AGENT_SELECTED" in source
    assert "AgentSelectionPayloadV1" in source
