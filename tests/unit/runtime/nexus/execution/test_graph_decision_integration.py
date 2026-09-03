# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionCriticAuthorityConflictError,
    DecisionFlowGateCapabilities,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import build_agent_execution_verification_pipeline
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = pytest.mark.unit


def _build_gate(*, contract) -> CanonicalDecisionFlowGate[AgentExecutionResult]:
    return CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=build_agent_execution_verification_pipeline(
                contract=contract,
            ),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )


def test_graph_without_decision_keeps_ordinary_behavior() -> None:
    executor = GraphExecutor(AgentRegistry())
    assert executor.peek_decision_flow_gate() is None
    assert executor.peek_critic_graph_hooks() is None


def test_decision_and_critic_authority_mutually_exclusive() -> None:
    registry = AgentRegistry()
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    registry.register(agent)
    contract = agent.get_contract()
    executor = GraphExecutor(registry)
    gate = _build_gate(contract=contract)
    hooks = MagicMock()
    hooks.config.verify_node_partial = False
    hooks.config.verify_graph_final = True
    hooks.config.verify_uaep_step = False
    hooks.verify_node_partial = False
    hooks.verify_graph_final = True
    executor.apply_decision_flow_gate(gate)
    with pytest.raises(DecisionCriticAuthorityConflictError):
        executor.apply_critic_graph_hooks(hooks)


def test_graph_executor_skips_critic_evaluator_loop_when_decision_active() -> None:
    registry = AgentRegistry()
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    registry.register(agent)
    contract = agent.get_contract()
    executor = GraphExecutor(registry)
    gate = _build_gate(contract=contract)
    executor.apply_decision_flow_gate(gate)
    assert executor._decision_authority_active() is True  # noqa: SLF001
