from __future__ import annotations

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_envelope import TaskEnvelope, routing_capability_from_envelope
from intergrax.runtime.registry.agent_routing_policy import evaluate_agent_routing
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.task.task import Task, TaskContext


class _StubAgent:
    def __init__(self, contract: AgentContract) -> None:
        self._contract = contract

    def get_contract(self) -> AgentContract:
        return self._contract

    async def run(self, request: AgentRunRequest) -> AgentRunResult:
        _ = request
        raise NotImplementedError("stub")

    def can_handle(self, task: TaskEnvelope) -> CapabilityMatchResult:
        capability = routing_capability_from_envelope(task) or ""
        if capability == "demo.basic":
            return CapabilityMatchResult(matched=True, score=1.0, rationale="stub")
        return CapabilityMatchResult(matched=False, score=0.0, rationale="no match")


def _contract(**updates: object) -> AgentContract:
    base = AgentContract(
        id="demo",
        name="Demo",
        description="demo",
        capabilities=["demo.basic"],
    )
    return base.model_copy(update=updates)


def test_retired_agent_is_not_routable() -> None:
    decision = evaluate_agent_routing(
        _contract(lifecycle_state=AgentLifecycleState.RETIRED),
        production_mode=False,
    )
    assert decision.routable is False


def test_production_mode_requires_owner_for_production_eligible_agent() -> None:
    decision = evaluate_agent_routing(
        _contract(production_eligible=True),
        production_mode=True,
    )
    assert decision.routable is False

    approved = evaluate_agent_routing(
        _contract(
            production_eligible=True,
            owner_team="platform",
            owner_contact="owner@intergrax",
            runbook_ref="docs/project/architecture/intergrax_runtime_architecture.md",
        ),
        production_mode=True,
    )
    assert approved.routable is True


def test_agent_router_skips_deprecated_agents() -> None:
    registry = AgentRegistry()
    active = _StubAgent(_contract(id="active"))
    retired = _StubAgent(
        _contract(id="retired", lifecycle_state=AgentLifecycleState.RETIRED)
    )
    registry.register(active)
    registry.register(retired)

    router = AgentRouter(registry)
    selected = router.route(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="hello",
            context=TaskContext(capability="demo.basic"),
        )
    )
    assert selected.get_contract().id == "active"


def test_agent_router_blocks_explicit_retired_agent() -> None:
    registry = AgentRegistry()
    retired = _StubAgent(
        _contract(id="retired", lifecycle_state=AgentLifecycleState.RETIRED)
    )
    registry.register(retired)
    router = AgentRouter(registry)

    with pytest.raises(RuntimeError, match="not routable"):
        router.route(
            Task(
                tenant_id="t1",
                user_id="u1",
                message="hello",
                agent_id="retired",
                context=TaskContext(),
            )
        )
