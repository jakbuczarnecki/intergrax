# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from intergrax.agents.authoring.decision_gateway import (
    ReflectionDecisionOutcome,
    decision_flow_gate_attached,
    resolve_decision_flow_gate,
    verify_reflection_draft_with_decision,
)
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowHostAction,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import build_agent_execution_verification_pipeline

pytestmark = pytest.mark.unit


def _contract() -> AgentContract:
    return AgentContract(
        id="reflection-agent",
        name="reflection-agent",
        description="test",
    )


def test_resolve_decision_flow_gate_from_metadata() -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=build_agent_execution_verification_pipeline(
                contract=_contract(),
            ),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
        ),
    )
    step_ctx = AgentStepContext(
        run_id=str(mint_run_id()),
        task_id=str(mint_task_id()),
        metadata={AcpRunContextKey.DECISION_FLOW_GATE: gate},
    )
    assert resolve_decision_flow_gate(step_ctx) is gate


@pytest.fixture
def execution_identity_binding():
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    yield
    reset_active_execution_identity(token)


@pytest.mark.asyncio
@patch("intergrax.agents.authoring.decision_gateway.evaluate_agent_execution_flow", new_callable=AsyncMock)
async def test_verify_reflection_draft_with_decision(mock_evaluate, execution_identity_binding) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=build_agent_execution_verification_pipeline(
                contract=_contract(),
            ),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
        ),
    )
    mock_evaluate.return_value = MagicMock(
        host_action=DecisionFlowHostAction.BLOCK,
        authority_reason="needs revision",
        revision_decision=None,
    )
    step_ctx = AgentStepContext(
        run_id=str(mint_run_id()),
        task_id=str(mint_task_id()),
        metadata={AcpRunContextKey.DECISION_FLOW_GATE: gate},
    )
    outcome = await verify_reflection_draft_with_decision(
        step_ctx,
        contract=_contract(),
        draft="draft text",
    )
    assert isinstance(outcome, ReflectionDecisionOutcome)
    assert outcome.passed is False
    assert outcome.host_action is DecisionFlowHostAction.BLOCK


def test_decision_flow_gate_attached() -> None:
    gate = MagicMock(spec=CanonicalDecisionFlowGate)
    step_ctx = AgentStepContext(
        run_id=str(mint_run_id()),
        task_id=str(mint_task_id()),
        metadata={AcpRunContextKey.DECISION_FLOW_GATE: gate},
    )
    assert decision_flow_gate_attached(step_ctx) is True
