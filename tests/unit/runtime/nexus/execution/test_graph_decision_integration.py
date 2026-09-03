# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_authorization import (
    DecisionGovernanceDecision,
    DecisionGovernanceDisposition,
    authoritative_decision_ref,
    decision_execution_action,
    decision_governance_policy_context,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewPending,
    governance_requires_human_review_reason,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionScope,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    DecisionArtifact,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import DecisionResolution
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_verification import VerificationDisposition
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionCriticAuthorityConflictError,
    DecisionFlowGateCapabilities,
    DecisionFlowGovernanceSpec,
    DecisionFlowHostAction,
    DecisionFlowIdentitySeed,
    DecisionFlowRequest,
    DecisionFlowScope,
    decision_identity_from_seed,
)
from intergrax.runtime.decision_flow_host import (
    build_agent_execution_verification_pipeline,
    decision_flow_result_to_validation_result,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

from tests.unit.runtime.test_decision_flow import (
    ChallengedStage,
    DenyGovernanceEvaluator,
    FailingHumanReviewPort,
    PassedStage,
    Payload,
    RecordingHumanReviewPort,
    RequireHumanGovernanceEvaluator,
    _governance_spec,
    _identity_seed,
    _pipeline,
    evaluator_spec_action,
    evaluator_spec_policy,
)

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


@pytest.fixture
def lifecycle_binding():
    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    yield
    reset_active_decision_lifecycle_host(token)


@pytest.mark.asyncio
async def test_graph_governance_deny_blocks_without_rejecting_accepted(
    lifecycle_binding,
) -> None:
    evaluator_spec = _governance_spec(
        DenyGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
            governance_spec=evaluator_spec,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    validation = decision_flow_result_to_validation_result(result)
    assert validation.valid is False
    assert result.accepted_decision is not None
    assert result.resolution_record is None
    assert result.authorization is None


@pytest.mark.asyncio
async def test_graph_governance_require_human_pending(lifecycle_binding) -> None:
    port = RecordingHumanReviewPort()
    evaluator_spec = _governance_spec(
        RequireHumanGovernanceEvaluator(
            action=evaluator_spec_action(),
            policy_context=evaluator_spec_policy(),
        ),
    )
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(PassedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
            governance_spec=evaluator_spec,
            human_review_port=port,
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="ok"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    assert result.host_action is DecisionFlowHostAction.PENDING_HUMAN
    assert result.accepted_decision is not None
    assert result.authorization is None


@pytest.mark.asyncio
async def test_graph_revision_required_blocks_without_terminal_resolution(
    lifecycle_binding,
) -> None:
    gate = CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=_pipeline(ChallengedStage(kind="test.stage")),
            revision_policy=decision_revision_policy(max_revisions=1),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )
    result = await gate.evaluate(
        DecisionFlowRequest(
            identity_seed=_identity_seed(),
            artifact_kind=validate_decision_artifact_kind("test.payload"),
            payload=Payload(text="bad"),
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        ),
    )
    validation = decision_flow_result_to_validation_result(result)
    assert validation.valid is False
    assert result.resolution_record is None
    assert result.lifecycle_state.stage is DecisionLifecycleStage.REVISION
