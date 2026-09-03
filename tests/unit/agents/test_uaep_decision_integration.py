# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
from intergrax.contracts.decision_authorization import (
    authoritative_decision_ref,
    decision_execution_action,
    decision_governance_policy_context,
    validate_decision_execution_action_kind,
)
from intergrax.contracts.decision_human_review import governance_requires_human_review_reason
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import validate_decision_artifact_kind
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
    DecisionCriticAuthorityConflictError,
    DecisionFlowGateCapabilities,
    DecisionFlowHostAction,
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import build_agent_execution_verification_pipeline
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task_contract import TaskExecutionOptions
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

from tests.unit.runtime.test_decision_flow import (
    ChallengedStage,
    DenyGovernanceEvaluator,
    PassedStage,
    Payload,
    RecordingHumanReviewPort,
    RequireHumanGovernanceEvaluator,
    _governance_spec,
    _pipeline,
    evaluator_spec_action,
    evaluator_spec_policy,
)

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


@pytest.fixture
def lifecycle_binding():
    token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
    yield
    reset_active_decision_lifecycle_host(token)


@pytest.fixture
def execution_identity_binding():
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    yield task_id, run_id
    reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_uaep_governance_deny_fails_without_rejecting_accepted(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id = execution_identity_binding
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    contract = agent.get_contract()
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
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
            governance_spec=evaluator_spec,
        ),
    )
    executor = UAEPExecutor()
    executor.set_decision_flow_gate(gate, verify_uaep_step=True)
    step = AgentStep(
        step_id=f"{contract.id}_step",
        step_name=f"{contract.id}_step",
        step_index=0,
        trace_label="cap.a",
    )
    step_result = StepExecutionResult(
        step_id=step.step_id,
        output=StepOutput(step_id=step.step_id, summary="ok", data={"value": "ok"}),
    )
    request = RuntimeRequest(
        agent_id=contract.id,
        user_id="user-1",
        session_id="sess-1",
        message="test",
        task_id=task_id,
        run_id=run_id,
        metadata={"tenant_id": "tenant-a"},
    )
    resolution = await executor._verify_uaep_step_authority(  # noqa: SLF001
        contract=contract,
        step=step,
        step_result=step_result,
        request=request,
        run_id=str(run_id),
        task_id=str(task_id),
        task_options=TaskExecutionOptions(),
        exec_ctx=MagicMock(),
    )
    assert resolution is not None
    assert resolution.agent_decision.type is AgentDecisionType.FAIL


@pytest.mark.asyncio
async def test_uaep_governance_require_human_requests_human(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id = execution_identity_binding
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    contract = agent.get_contract()
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
            scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
            governance_spec=evaluator_spec,
            human_review_port=port,
        ),
    )
    executor = UAEPExecutor()
    executor.set_decision_flow_gate(gate, verify_uaep_step=True)
    step = AgentStep(
        step_id=f"{contract.id}_step",
        step_name=f"{contract.id}_step",
        step_index=0,
        trace_label="cap.a",
    )
    step_result = StepExecutionResult(
        step_id=step.step_id,
        output=StepOutput(step_id=step.step_id, summary="ok", data={"value": "ok"}),
    )
    request = RuntimeRequest(
        agent_id=contract.id,
        user_id="user-1",
        session_id="sess-1",
        message="test",
        task_id=task_id,
        run_id=run_id,
        metadata={"tenant_id": "tenant-a"},
    )
    resolution = await executor._verify_uaep_step_authority(  # noqa: SLF001
        contract=contract,
        step=step,
        step_result=step_result,
        request=request,
        run_id=str(run_id),
        task_id=str(task_id),
        task_options=TaskExecutionOptions(),
        exec_ctx=MagicMock(),
    )
    assert resolution is not None
    assert resolution.agent_decision.type is AgentDecisionType.REQUEST_HUMAN
    assert port.pending is not None
    assert port.pending.request.reason_code == governance_requires_human_review_reason()
