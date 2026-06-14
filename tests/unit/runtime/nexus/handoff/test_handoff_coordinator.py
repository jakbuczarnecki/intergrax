# © Artur Czarnecki. All rights reserved.

from intergrax.utils import attribute_access
import pytest

from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_handoff import AgentHandoff, handoff_from_decision
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.nexus.handoff.coordinator import HandoffCoordinator
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.agents.agent_contract import Agent
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager


class _StubAgent(Agent):
    def __init__(self, *, agent_id: str, capability: str) -> None:
        self._agent_id = agent_id
        self._capability = capability

    def get_contract(self) -> AgentContract:
        return AgentContract(
            id=self._agent_id,
            name=self._agent_id,
            description="handoff stub",
            capabilities=[self._capability],
        )

    def can_handle(self, task_context: object) -> CapabilityMatchResult:
        capability = attribute_access.optional(task_context, "capability", None)
        if capability == self._capability:
            return CapabilityMatchResult(
                matched=True,
                agent_id=self._agent_id,
                matched_capabilities=[self._capability],
                score=1.0,
            )
        return CapabilityMatchResult(matched=False)

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(fixed_text="ok"),
            enable_rag=False,
            production_mode=False,
            tenant_id=request.tenant_id,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )


def _registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(_StubAgent(agent_id="agent_a", capability="cap.a"))
    registry.register(_StubAgent(agent_id="agent_b", capability="cap.b"))
    return registry


@pytest.mark.unit
@pytest.mark.gate
def test_agent_handoff_requires_target():
    with pytest.raises(ValueError, match="requires to_agent_id or to_capability"):
        AgentHandoff(from_agent_id="agent_a")


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_from_decision_payload():
    decision = AgentDecision(
        type=AgentDecisionType.MODIFY_PLAN,
        reason="delegate research",
        payload={
            "handoff": {
                "from_agent_id": "agent_a",
                "to_capability": "cap.b",
                "reason": "needs specialist",
                "payload": {"topic": "vendors"},
            }
        },
    )
    handoff = handoff_from_decision(decision)
    assert handoff is not None
    assert handoff.to_capability == "cap.b"
    assert handoff.payload["topic"] == "vendors"


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_coordinator_resolves_capability():
    coordinator = HandoffCoordinator(_registry())
    handoff = AgentHandoff(
        from_agent_id="agent_a",
        to_capability="cap.b",
        reason="delegate",
    )
    result = coordinator.validate(handoff, from_agent_id="agent_a")
    assert result.valid is True
    assert result.resolved_agent_id == "agent_b"


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_coordinator_apply_to_graph():
    coordinator = HandoffCoordinator(_registry())
    graph = ExecutionGraph(graph_id="g1", task_id="task_1", nodes=[
        ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.a"),
    ])
    handoff = AgentHandoff(
        from_agent_id="agent_a",
        to_agent_id="agent_b",
        reason="delegate",
    )
    node = coordinator.apply_to_graph(
        graph,
        handoff,
        from_node_id="n1",
        resolved_agent_id="agent_b",
    )
    assert node.node_id.startswith("handoff_")
    assert node.agent_id == "agent_b"
    assert node.depends_on == ["n1"]
    assert len(graph.nodes) == 2


@pytest.mark.unit
@pytest.mark.gate
def test_handoff_coordinator_rejects_unknown_target():
    coordinator = HandoffCoordinator(_registry())
    handoff = AgentHandoff(
        from_agent_id="agent_a",
        to_agent_id="missing_agent",
        reason="delegate",
    )
    result = coordinator.validate(handoff, from_agent_id="agent_a")
    assert result.valid is False
    assert any("unknown target" in err for err in result.errors)


@pytest.mark.unit
@pytest.mark.gate
def test_resolve_handoff_from_execution():
    from intergrax.contracts.agent_handoff import resolve_handoff_from_execution

    handoff = AgentHandoff(from_agent_id="agent_a", to_agent_id="agent_b", reason="x")
    execution = AgentExecutionResult(
        agent_id="agent_a",
        run_id="run_1",
        status=AgentExecutionStatus.COMPLETED,
        summary="done",
        agent_decision=AgentDecision(
            type=AgentDecisionType.MODIFY_PLAN,
            reason="handoff",
            handoff=handoff,
        ),
    )
    assert resolve_handoff_from_execution(execution) == handoff
