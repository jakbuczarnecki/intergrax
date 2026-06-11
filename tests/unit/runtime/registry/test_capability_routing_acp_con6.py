# © Artur Czarnecki. All rights reserved.

"""ACP-CON-6 — capability token routing without class names in task payload."""

from __future__ import annotations

import pytest

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.task_routing import TaskRoutingViolationError, validate_task_routing_payload
from intergrax.runtime.nexus.agent_router import AgentRouter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.registry.capability_routing import select_best_capability_match
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager

_SHARED_CAPABILITY = "routing.demo.cap"


def _contract(agent_id: str) -> AgentContract:
    return AgentContract(
        id=agent_id,
        name=agent_id,
        description="routing demo",
        capabilities=[_SHARED_CAPABILITY],
        risk_level=AgentRiskLevel.LOW,
        max_steps=3,
    )


class _RoutingDemoAgent(Agent):
    def __init__(self, agent_id: str, score: float) -> None:
        self._agent_id = agent_id
        self._score = score

    def get_contract(self) -> AgentContract:
        return _contract(self._agent_id)

    def can_handle(self, task_context: TaskContext) -> CapabilityMatchResult:
        _ = task_context
        return CapabilityMatchResult(
            matched=True,
            agent_id=self._agent_id,
            matched_capabilities=[_SHARED_CAPABILITY],
            score=self._score,
            rationale="demo",
        )

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        _ = request
        config = RuntimeConfig(llm_adapter=FakeLLMAdapter(), production_mode=False)
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_validate_task_routing_rejects_agent_class_metadata() -> None:
    with pytest.raises(TaskRoutingViolationError, match="agent_class"):
        validate_task_routing_payload(metadata={"agent_class": "EchoAgent"})


@pytest.mark.unit
@pytest.mark.gate
def test_validate_task_routing_rejects_import_path() -> None:
    with pytest.raises(TaskRoutingViolationError, match="agent_import_path"):
        validate_task_routing_payload(
            context_metadata={"agent_import_path": "echo.echo_agent.EchoAgent"},
        )


@pytest.mark.unit
@pytest.mark.gate
def test_select_best_capability_match_prefers_higher_score() -> None:
    registry = AgentRegistry()
    registry.register(_RoutingDemoAgent("routing-demo-a", 1.0))
    registry.register(_RoutingDemoAgent("routing-demo-b", 2.5))
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="route by capability",
        context=TaskContext(capability=_SHARED_CAPABILITY),
    )
    route = select_best_capability_match(registry, task, _SHARED_CAPABILITY)
    assert route.selected is not None
    assert route.selected.get_contract().id == "routing-demo-b"
    assert route.selection_reason == "capability_best_score"


@pytest.mark.unit
@pytest.mark.gate
def test_agent_router_rejects_class_name_in_task_metadata() -> None:
    registry = AgentRegistry()
    registry.register(_RoutingDemoAgent("routing-demo-a", 1.0))
    router = AgentRouter(registry)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="bad routing",
        context=TaskContext(capability=_SHARED_CAPABILITY),
        metadata={"required_agent_class": "EchoAgent"},
    )
    with pytest.raises(TaskRoutingViolationError):
        router.route(task)
