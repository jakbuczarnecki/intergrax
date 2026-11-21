"""FC-1 Part A: AgentRouter posture is explicit and the first-routable fallback is typed LAB opt-in.

Negative tests NT-FC1-05..11 of T-IGX-043-A (EQOA-A4-009, A4-015a, DP-26, DP-27).
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import Token

import pytest
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.task_envelope import TaskEnvelope, routing_capability_from_envelope
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus import agent_router
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = pytest.mark.unit


class _NoMatchAgent:
    """Routable agent whose ``can_handle`` never matches a task without a known capability."""

    def __init__(self, contract: AgentContract) -> None:
        self._contract = contract

    def get_contract(self) -> AgentContract:
        return self._contract

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        return AgentRunResult(output="ok")

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        if routing_capability_from_envelope(task) == "demo.basic":
            return CapabilityMatchResult(matched=True, score=0.9, reason="stub")
        return CapabilityMatchResult(matched=False, score=0.0, reason="no match")


def _contract(agent_id: str = "demo") -> AgentContract:
    return AgentContract(
        id=agent_id,
        name=agent_id,
        description=agent_id,
        capabilities=["demo.basic"],
        max_steps=1,
    )


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(_NoMatchAgent(_contract("first")))
    registry.register(_NoMatchAgent(_contract("second")))
    return registry


def _task_without_match(**updates: object) -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="no capability named",
        context=TaskContext(),
        **updates,
    )


def _bound_identity() -> Token:
    return bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )


def _construct(factory: Callable[..., object], *args: object) -> object:
    """Call a constructor positionally so the missing keyword is a runtime TypeError."""
    return factory(*args)


def test_agent_router_requires_explicit_production_mode() -> None:
    """NT-FC1-05."""
    with pytest.raises(TypeError):
        _construct(AgentRouter, AgentRegistry())


def test_nexus_loop_requires_explicit_production_mode() -> None:
    """NT-FC1-06."""
    with pytest.raises(TypeError):
        _construct(NexusLoop, AgentRegistry())


def test_strict_router_refuses_first_routable_fallback() -> None:
    """NT-FC1-07: STRICT/PRODUCT never picks the first routable agent; no event is emitted."""
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    router = AgentRouter(_registry(), production_mode=True, event_bus=bus)
    task = _task_without_match()
    token = _bound_identity()
    try:
        with pytest.raises(agent_router.AgentRoutingError):
            router.route(task)
    finally:
        reset_active_execution_identity(token)
    assert store.list_for_task(task.task_id, tenant_id="t1") == []


def test_lab_router_without_opt_in_refuses_fallback() -> None:
    """NT-FC1-08: LAB default is deny as well; the fallback needs the typed opt-in."""
    router = AgentRouter(_registry(), production_mode=False)
    with pytest.raises(agent_router.AgentRoutingError):
        router.route(_task_without_match())


def test_lab_opt_in_keeps_fallback_and_emits_fallback_used() -> None:
    """NT-FC1-09."""
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    router = AgentRouter(
        _registry(),
        production_mode=False,
        event_bus=bus,
        lab_first_routable_fallback=True,
    )
    task = _task_without_match()
    token = _bound_identity()
    try:
        selected = router.route(task)
    finally:
        reset_active_execution_identity(token)
    assert selected.get_contract().id == "first"
    events = store.list_for_task(task.task_id, tenant_id="t1")
    assert len(events) == 1
    assert events[0].event_type == RuntimeEventType.AGENT_SELECTED
    data = events[0].payload["data"]
    assert data["fallback_used"] is True
    assert data["selection_reason"] == "fallback_first_routable"


def test_strict_router_with_lab_fallback_opt_in_is_refused_at_construction() -> None:
    """NT-FC1-10: contradictory posture, no silent downgrade."""
    with pytest.raises(ValueError):
        AgentRouter(
            AgentRegistry(),
            production_mode=True,
            lab_first_routable_fallback=True,
        )


def test_explicit_agent_id_and_capability_paths_unchanged() -> None:
    """NT-FC1-11 (control): explicit agent_id and capability routes behave as before."""
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store)
    router = AgentRouter(_registry(), production_mode=True, event_bus=bus)

    explicit = Task(
        tenant_id="t1",
        user_id="u1",
        message="explicit",
        agent_id="second",
        context=TaskContext(),
    )
    by_capability = Task(
        tenant_id="t1",
        user_id="u1",
        message="by capability",
        context=TaskContext(capability="demo.basic"),
    )
    token = _bound_identity()
    try:
        assert router.route(explicit).get_contract().id == "second"
        capability_agent = router.route(by_capability)
    finally:
        reset_active_execution_identity(token)
    assert capability_agent.get_contract().id in {"first", "second"}
    explicit_events = store.list_for_task(explicit.task_id, tenant_id="t1")
    assert [e.payload["data"]["selection_reason"] for e in explicit_events] == ["explicit_agent_id"]
    capability_events = store.list_for_task(by_capability.task_id, tenant_id="t1")
    assert len(capability_events) == 1
    assert capability_events[0].payload["data"]["fallback_used"] is False
