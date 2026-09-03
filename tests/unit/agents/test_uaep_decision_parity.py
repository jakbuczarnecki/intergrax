# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from intergrax.agents.uaep import UAEPExecutor
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.agent_step import AgentStep, StepExecutionResult, StepOutput
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
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.migration.critic_shadow_adapter import (
    build_critic_shadow_adapter,
    observe_uaep_step_parity,
)
from intergrax.runtime.migration.decision_critic_parity import (
    DecisionCriticParityClassification,
    DecisionCriticParityResult,
)
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task_contract import TaskExecutionOptions
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = pytest.mark.unit


@dataclass
class _RecordingParityObserver:
    results: list[DecisionCriticParityResult] = field(default_factory=list)

    def record(self, result: DecisionCriticParityResult) -> None:
        self.results.append(result)


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
    yield task_id, run_id, attempt_id
    reset_active_execution_identity(token)


@pytest.mark.asyncio
async def test_uaep_shadow_observation_does_not_change_decision_resolution(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    agent = UaepPipelineStubAgent(agent_id="agent-a", capability="cap.a")
    contract = agent.get_contract()
    gate = _build_gate(contract=contract)
    executor = UAEPExecutor()
    executor.set_decision_flow_gate(gate, verify_uaep_step=True)
    shadow = build_critic_shadow_adapter()
    observer = _RecordingParityObserver()
    executor.set_critic_parity_shadow(shadow, observer=observer)
    step = AgentStep(
        step_id=f"{contract.id}_step",
        step_name=f"{contract.id}_step",
        step_index=0,
        trace_label="cap.a",
    )
    step_result = StepExecutionResult(
        step_id=step.step_id,
        output=StepOutput(step_id=step.step_id, summary="valid summary", data={}),
    )
    request = RuntimeRequest(
        agent_id=contract.id,
        user_id="user-1",
        session_id="sess-1",
        message="test",
        task_id=str(task_id),
        run_id=str(run_id),
        tenant_id="tenant-1",
    )
    exec_ctx = MagicMock()
    exec_ctx.task_id = str(task_id)
    exec_ctx.metadata = {}
    baseline = await executor._verify_uaep_step_decision_flow(  # noqa: SLF001
        contract=contract,
        step=step,
        step_result=step_result,
        request=request,
        run_id=str(run_id),
        task_id=str(task_id),
        task_options=TaskExecutionOptions(),
        exec_ctx=exec_ctx,
    )
    shadow.orchestrator.verify = lambda request, *, contract=None: CriticVerdict(  # type: ignore[method-assign]
        scope=CriticScope.UAEP_STEP,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                score=0.0,
                errors=["fail"],
            ),
        ],
        recommended_action=CriticAction.FAIL,
        failure_reasons=("fail",),
    )
    after_fail_shadow = await executor._verify_uaep_step_decision_flow(  # noqa: SLF001
        contract=contract,
        step=step,
        step_result=step_result,
        request=request,
        run_id=str(run_id),
        task_id=str(task_id),
        task_options=TaskExecutionOptions(),
        exec_ctx=exec_ctx,
    )
    assert baseline == after_fail_shadow
    assert observer.results[-1].classification is DecisionCriticParityClassification.MISMATCH


@pytest.mark.asyncio
async def test_uaep_step_qualification_matrix(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    task_id, run_id, attempt_id = execution_identity_binding
    contract = AgentContract(
        id="agent-a",
        name="agent-a",
        description="parity",
        validation_rules=["non_empty_summary"],
    )
    gate = _build_gate(contract=contract)
    shadow = build_critic_shadow_adapter()
    execution = AgentExecutionResult(
        agent_id="agent-a",
        run_id=str(run_id),
        status=AgentExecutionStatus.COMPLETED,
        summary="valid summary",
    )
    from intergrax.runtime.decision_flow_host import (
        agent_execution_decision_context,
        agent_execution_identity_seed,
        build_agent_execution_flow_request,
        evaluate_agent_execution_flow,
    )

    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="uaep.step",
        subject="step-1",
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.UAEP_STEP,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    result = await observe_uaep_step_parity(
        shadow=shadow,
        decision_result=flow_result,
        execution=execution,
        contract=contract,
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id="tenant-1",
        step_id="step-1",
    )
    assert result.classification is DecisionCriticParityClassification.CAPABILITY_GAP
    assert flow_result.host_action is DecisionFlowHostAction.CONTINUE
