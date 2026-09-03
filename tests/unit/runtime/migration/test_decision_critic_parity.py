# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    candidate_decision,
    candidate_decision_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionDecision,
    DecisionRevisionDisposition,
    decision_revision_policy,
)
from intergrax.contracts.decision_verification import (
    VerificationDisposition,
    VerificationStageOutcome,
    validate_verification_finding_code,
    validate_verification_requirement_code,
    validate_verification_stage_kind,
    verification_challenge,
    verification_finding,
    verification_result,
    verification_stage_record,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
from intergrax.runtime.decision_flow import (
    DecisionFlowHostAction,
    DecisionFlowResult,
    DecisionFlowScope,
)
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    DecisionCriticParityClassification,
    DECISION_ACCEPT_CRITIC_CHALLENGE,
    LEGACY_L2_NOT_DECISION_VERIFICATION,
    LEGACY_RETRY_IS_EXECUTION_RETRY,
    LEGACY_REVISE_IS_DECISION_REVISION,
    ParityHostScope,
    ParityVerificationCapability,
    aggregate_parity_metrics,
    build_parity_identity,
    compare_decision_critic_parity,
    evaluate_critic_retirement_readiness,
    project_critic_observation,
    project_decision_observation,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_MODULE_PATHS = (
    Path("intergrax/runtime/migration/decision_critic_parity.py"),
    Path("intergrax/runtime/migration/critic_shadow_adapter.py"),
)
_FORBIDDEN_FRAGMENTS = (
    "Any",
    "cast(",
    "type: ignore",
    "pyright: ignore",
    "getattr",
    "setattr",
    "hasattr",
    "inspect.",
    "exec(",
    "eval(",
    "dict[str, Any]",
)


@dataclass(frozen=True, slots=True)
class _Payload:
    text: str


def _build_candidate(summary: str = "ok"):
    artifact_kind = validate_decision_artifact_kind("agent.execution.result")
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="parity", subject="subject-1"),
        tenant_id="tenant-1",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        ),
    )
    return candidate_decision(
        identity=identity,
        artifact_kind=artifact_kind,
        payload=_Payload(text=summary),
    )


def _decision_result(
    *,
    disposition: VerificationDisposition,
    host_action: DecisionFlowHostAction,
    revision_disposition: DecisionRevisionDisposition | None = None,
) -> DecisionFlowResult[_Payload]:
    candidate = _build_candidate()
    proposal_ref = candidate_decision_ref(candidate)
    stage_outcome = (
        VerificationStageOutcome.PASSED
        if disposition is VerificationDisposition.PASSED
        else VerificationStageOutcome.CHALLENGED
    )
    stage_kind = validate_verification_stage_kind("structural")
    if stage_outcome is VerificationStageOutcome.PASSED:
        stage_record = verification_stage_record(
            proposal_ref=proposal_ref,
            stage=stage_kind,
            outcome=stage_outcome,
        )
    else:
        finding = verification_finding(
            code=validate_verification_finding_code("verification.test.challenged"),
            message="challenged",
        )
        stage_record = verification_stage_record(
            proposal_ref=proposal_ref,
            stage=stage_kind,
            outcome=stage_outcome,
            challenge=verification_challenge(
                proposal_ref=proposal_ref,
                stage=stage_kind,
                requirement_code=validate_verification_requirement_code(
                    "verification.structural.agent_execution",
                ),
                finding=finding,
            ),
        )
    verification = verification_result(
        proposal_ref=proposal_ref,
        disposition=disposition,
        stage_records=(stage_record,),
    )
    revision = None
    if revision_disposition is not None:
        revision = DecisionRevisionDecision(
            disposition=revision_disposition,
            proposal_ref=candidate_decision_ref(candidate),
            policy=decision_revision_policy(max_revisions=1),
            revision_number=1,
        )
    lifecycle_host = CanonicalDecisionLifecycleHost()
    lifecycle_state = lifecycle_host.start(candidate.identity)
    return DecisionFlowResult(
        host_action=host_action,
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        candidate=candidate,
        verification_result=verification,
        lifecycle_state=lifecycle_state,
        revision_decision=revision,
    )


def _critic_verdict(*, passed: bool, action: CriticAction = CriticAction.CONTINUE) -> CriticVerdict:
    layer = LayerVerdict(layer=CriticLayer.L0_DETERMINISTIC, passed=passed, score=1.0 if passed else 0.0)
    return CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=passed,
        layers=[layer],
        recommended_action=action,
        failure_reasons=() if passed else ("structural failure",),
    )


def test_project_decision_acceptable_on_passed_continue() -> None:
    result = _decision_result(
        disposition=VerificationDisposition.PASSED,
        host_action=DecisionFlowHostAction.CONTINUE,
    )
    observation = project_decision_observation(result)
    assert observation.outcome.value == "acceptable"


def test_project_critic_acceptable_on_pass() -> None:
    observation = project_critic_observation(_critic_verdict(passed=True))
    assert observation.outcome.value == "acceptable"


def test_compare_match_on_aligned_outcomes() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.PASSED,
            host_action=DecisionFlowHostAction.CONTINUE,
        ),
        critic_verdict=_critic_verdict(passed=True),
    )
    assert parity.classification is DecisionCriticParityClassification.MATCH
    assert parity.retirement_blocking is False


def test_compare_expected_difference_on_legacy_retry() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.CHALLENGED,
            host_action=DecisionFlowHostAction.BLOCK,
        ),
        critic_verdict=_critic_verdict(passed=False, action=CriticAction.RETRY),
    )
    assert parity.classification is DecisionCriticParityClassification.EXPECTED_DIFFERENCE
    assert LEGACY_RETRY_IS_EXECUTION_RETRY in {item.code for item in parity.differences}
    assert parity.retirement_blocking is False


def test_compare_expected_difference_on_legacy_revise_with_revision_allowed() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.CHALLENGED,
            host_action=DecisionFlowHostAction.BLOCK,
            revision_disposition=DecisionRevisionDisposition.ALLOWED,
        ),
        critic_verdict=_critic_verdict(passed=False, action=CriticAction.REVISE),
    )
    assert parity.classification is DecisionCriticParityClassification.EXPECTED_DIFFERENCE
    assert LEGACY_REVISE_IS_DECISION_REVISION in {item.code for item in parity.differences}


def test_compare_blocking_mismatch_decision_accept_critic_challenge() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.PASSED,
            host_action=DecisionFlowHostAction.CONTINUE,
        ),
        critic_verdict=_critic_verdict(passed=False, action=CriticAction.FAIL),
    )
    assert parity.classification is DecisionCriticParityClassification.MISMATCH
    assert DECISION_ACCEPT_CRITIC_CHALLENGE in {item.code for item in parity.differences}
    assert parity.retirement_blocking is True


def test_compare_shadow_unavailable() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.PASSED,
            host_action=DecisionFlowHostAction.CONTINUE,
        ),
        shadow_unavailable=True,
    )
    assert parity.classification is DecisionCriticParityClassification.SHADOW_UNAVAILABLE
    assert parity.retirement_blocking is False


def test_compare_shadow_error() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.PASSED,
            host_action=DecisionFlowHostAction.CONTINUE,
        ),
        shadow_error="shadow exploded",
    )
    assert parity.classification is DecisionCriticParityClassification.SHADOW_ERROR


def test_compare_expected_difference_on_l2_hitl() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    verdict = CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=False,
        layers=[
            LayerVerdict(layer=CriticLayer.L2_HUMAN, passed=False, errors=["human required"]),
        ],
        recommended_action=CriticAction.ESCALATE_HITL,
        failure_reasons=("human required",),
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.CHALLENGED,
            host_action=DecisionFlowHostAction.PENDING_HUMAN,
        ),
        critic_verdict=verdict,
    )
    assert parity.classification is DecisionCriticParityClassification.EXPECTED_DIFFERENCE
    assert LEGACY_L2_NOT_DECISION_VERIFICATION in {item.code for item in parity.differences}


def test_retirement_ready_with_required_evidence() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    uaep_identity = build_parity_identity(
        flow_scope=DecisionFlowScope.UAEP_STEP,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="step-1",
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
            ),
            critic_verdict=_critic_verdict(passed=True),
        ),
        compare_decision_critic_parity(
            identity=uaep_identity,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
            ),
            critic_verdict=_critic_verdict(passed=True),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL, ParityHostScope.UAEP_STEP}),
        required_capabilities=frozenset({ParityVerificationCapability.STRUCTURAL}),
    )
    assert report.readiness is CriticRetirementReadiness.READY


def test_retirement_not_ready_on_blocking_mismatch() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
            ),
            critic_verdict=_critic_verdict(passed=False, action=CriticAction.FAIL),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        required_capabilities=frozenset({ParityVerificationCapability.STRUCTURAL}),
    )
    assert report.readiness is CriticRetirementReadiness.NOT_READY


def test_retirement_insufficient_evidence_when_uaep_missing() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
            ),
            critic_verdict=_critic_verdict(passed=True),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL, ParityHostScope.UAEP_STEP}),
        required_capabilities=frozenset({ParityVerificationCapability.STRUCTURAL}),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityHostScope.UAEP_STEP in report.missing_scopes


def test_aggregate_parity_metrics() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
            ),
            critic_verdict=_critic_verdict(passed=True),
        ),
        compare_decision_critic_parity(
            identity=identity,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
            ),
            shadow_unavailable=True,
        ),
    ]
    metrics = aggregate_parity_metrics(results)
    assert metrics.total_comparisons == 2
    assert metrics.matches == 1
    assert metrics.shadow_unavailable == 1


def test_forbidden_audit_migration_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
