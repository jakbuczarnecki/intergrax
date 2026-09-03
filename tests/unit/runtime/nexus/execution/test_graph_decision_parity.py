# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
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
    DecisionFlowScope,
)
from intergrax.runtime.decision_flow_host import (
    build_agent_execution_flow_request,
    build_agent_execution_verification_pipeline,
    decision_flow_result_to_validation_result,
    evaluate_agent_execution_flow,
    agent_execution_decision_context,
    agent_execution_identity_seed,
)
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.migration.critic_shadow_adapter import (
    CriticShadowAdapter,
    CriticShadowConfig,
    build_critic_shadow_adapter,
    observe_graph_final_parity,
)
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    DecisionCriticParityClassification,
    DecisionCriticParityResult,
    ParityCapabilityRequirement,
    ParityCapabilityRequirementMode,
    ParityHostScope,
    ParityVerificationCapability,
    aggregate_parity_metrics,
    evaluate_critic_retirement_readiness,
)
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)

pytestmark = pytest.mark.unit


@dataclass
class _RecordingParityObserver:
    results: list[DecisionCriticParityResult] = field(default_factory=list)

    def record(self, result: DecisionCriticParityResult) -> None:
        self.results.append(result)


def _structural_contract() -> AgentContract:
    return AgentContract(
        id="agent-a",
        name="agent-a",
        description="parity",
        validation_rules=["non_empty_summary"],
    )


def _build_gate(*, contract: AgentContract) -> CanonicalDecisionFlowGate[AgentExecutionResult]:
    return CanonicalDecisionFlowGate(
        capabilities=DecisionFlowGateCapabilities(
            verification_pipeline=build_agent_execution_verification_pipeline(
                contract=contract,
            ),
            revision_policy=decision_revision_policy(max_revisions=0),
            scopes=frozenset({DecisionFlowScope.GRAPH_FINAL}),
        ),
    )


def _execution(*, summary: str) -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id="agent-a",
        run_id="run-1",
        status=AgentExecutionStatus.COMPLETED,
        summary=summary,
    )


def _pass_verdict() -> CriticVerdict:
    return CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=True,
        layers=[LayerVerdict(layer=CriticLayer.L0_DETERMINISTIC, passed=True, score=1.0)],
        recommended_action=CriticAction.CONTINUE,
    )


def _fail_verdict() -> CriticVerdict:
    return CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                score=0.0,
                errors=["structural failure"],
            ),
        ],
        recommended_action=CriticAction.FAIL,
        failure_reasons=("structural failure",),
    )


def _stub_shadow(adapter: CriticShadowAdapter, verdict: CriticVerdict) -> None:
    def _verify_final(request, *, contract=None):
        return verdict

    adapter.orchestrator.verify_final = _verify_final  # type: ignore[method-assign]


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
async def test_graph_production_validation_invariant_across_shadow_modes(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    contract = _structural_contract()
    gate = _build_gate(contract=contract)
    task_id, run_id, attempt_id = execution_identity_binding
    execution = _execution(summary="valid summary")
    decision_context = agent_execution_decision_context(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        tenant_id="tenant-1",
    )
    identity_seed = agent_execution_identity_seed(
        context=decision_context,
        namespace="graph.final",
        subject="graph-1",
    )
    flow_request = build_agent_execution_flow_request(
        execution=execution,
        identity_seed=identity_seed,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
    )
    flow_result = await evaluate_agent_execution_flow(gate, flow_request)
    baseline_validation = decision_flow_result_to_validation_result(flow_result)
    observer = _RecordingParityObserver()
    shadow = build_critic_shadow_adapter()
    for verdict in (_pass_verdict(), _fail_verdict()):
        _stub_shadow(shadow, verdict)
        await observe_graph_final_parity(
            shadow=shadow,
            decision_result=flow_result,
            execution=execution,
            contract=contract,
            task_id=str(task_id),
            run_id=str(run_id),
            attempt_id=str(attempt_id),
            tenant_id="tenant-1",
            graph_id="graph-1",
            observer=observer,
        )
        assert decision_flow_result_to_validation_result(flow_result) == baseline_validation
    broken = build_critic_shadow_adapter()

    def _explode(request, *, contract=None):
        raise RuntimeError("shadow exploded")

    broken.orchestrator.verify_final = _explode  # type: ignore[method-assign]
    await observe_graph_final_parity(
        shadow=broken,
        decision_result=flow_result,
        execution=execution,
        contract=contract,
        task_id=str(task_id),
        run_id=str(run_id),
        attempt_id=str(attempt_id),
        tenant_id="tenant-1",
        graph_id="graph-1",
        observer=observer,
    )
    assert decision_flow_result_to_validation_result(flow_result) == baseline_validation
    assert observer.results[-1].classification is DecisionCriticParityClassification.SHADOW_ERROR


@pytest.mark.asyncio
async def test_qualification_matrix_structural_clean_and_failure(
    lifecycle_binding,
    execution_identity_binding,
) -> None:
    contract = _structural_contract()
    gate = _build_gate(contract=contract)
    shadow = build_critic_shadow_adapter(config=CriticShadowConfig())
    task_id, run_id, attempt_id = execution_identity_binding
    cases = (
        ("clean", "valid summary", DecisionCriticParityClassification.CAPABILITY_GAP),
        ("structural_failure", "", DecisionCriticParityClassification.MATCH),
    )
    results: list[DecisionCriticParityResult] = []
    for _case_id, summary, expected in cases:
        execution = _execution(summary=summary)
        decision_context = agent_execution_decision_context(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            tenant_id="tenant-1",
        )
        identity_seed = agent_execution_identity_seed(
            context=decision_context,
            namespace="graph.final",
            subject="graph-1",
        )
        flow_request = build_agent_execution_flow_request(
            execution=execution,
            identity_seed=identity_seed,
            flow_scope=DecisionFlowScope.GRAPH_FINAL,
        )
        flow_result = await evaluate_agent_execution_flow(gate, flow_request)
        result = await observe_graph_final_parity(
            shadow=shadow,
            decision_result=flow_result,
            execution=execution,
            contract=contract,
            task_id=str(task_id),
            run_id=str(run_id),
            attempt_id=str(attempt_id),
            tenant_id="tenant-1",
            graph_id="graph-1",
        )
        results.append(result)
        assert result.classification is expected
    metrics = aggregate_parity_metrics(results)
    assert metrics.total_comparisons == 2
    readiness = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=(
            ParityCapabilityRequirement(
                ParityVerificationCapability.STRUCTURAL,
                ParityCapabilityRequirementMode.CROSS_SYSTEM,
            ),
        ),
    )
    assert readiness.readiness in {
        CriticRetirementReadiness.READY,
        CriticRetirementReadiness.NOT_READY,
        CriticRetirementReadiness.INSUFFICIENT_EVIDENCE,
    }
