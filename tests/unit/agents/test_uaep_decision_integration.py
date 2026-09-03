# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionCriticAuthorityConflictError,
    DecisionFlowGateCapabilities,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import build_agent_execution_verification_pipeline
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = pytest.mark.unit


def _build_gate(*, contract) -> CanonicalDecisionFlowGate:
    return CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=build_agent_execution_verification_pipeline(
                contract=contract,
            ),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
        ),
    )


def test_uaep_without_decision_keeps_ordinary_behavior() -> None:
    executor = UAEPExecutor()
    assert executor._decision_flow_gate is None  # noqa: SLF001


def test_uaep_decision_and_critic_authority_mutually_exclusive() -> None:
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    contract = agent.get_contract()
    executor = UAEPExecutor()
    gate = _build_gate(contract=contract)
    hooks = MagicMock()
    hooks.config.verify_node_partial = False
    hooks.config.verify_graph_final = False
    hooks.config.verify_uaep_step = True
    executor.set_decision_flow_gate(gate, verify_uaep_step=True)
    with pytest.raises(DecisionCriticAuthorityConflictError):
        executor.set_critic_hooks(hooks, verify_uaep_step=True)


def test_same_gate_contract_used_by_graph_and_uaep() -> None:
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    contract = agent.get_contract()
    gate = _build_gate(contract=contract)
    graph_gate = gate
    uaep_gate = gate
    assert graph_gate is uaep_gate
    assert graph_gate.capabilities.scopes != frozenset()
