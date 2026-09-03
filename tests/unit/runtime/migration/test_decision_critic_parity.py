# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewPending,
    decision_human_review_request,
    verification_challenged_human_review_reason,
)
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
from intergrax.runtime.decision_human_review import request_decision_human_review
from intergrax.runtime.migration.legacy_critic_human_evidence import (
    LegacyCriticHumanEscalationEvidence,
    proven_retired_l2_human_escalation_evidence,
)
from intergrax.runtime.migration.decision_critic_parity import (
    CriticRetirementReadiness,
    DecisionCriticParityClassification,
    DECISION_ACCEPT_CRITIC_CHALLENGE,
    CRITIC_CAPABILITY_NOT_EXERCISED_BY_DECISION,
    DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    LEGACY_HITL_IS_DECISION_HUMAN_REVIEW,
    LEGACY_L2_NOT_DECISION_VERIFICATION,
    LEGACY_RETRY_IS_EXECUTION_RETRY,
    LEGACY_REVISE_IS_DECISION_REVISION,
    ParityCapabilityRequirement,
    ParityCapabilityRequirementMode,
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
    Path("intergrax/runtime/migration/legacy_critic_human_evidence.py"),
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
    stage_kinds: tuple[str, ...] = ("structural",),
    human_review_pending: DecisionHumanReviewPending | None = None,
) -> DecisionFlowResult[_Payload]:
    candidate = _build_candidate()
    proposal_ref = candidate_decision_ref(candidate)
    stage_outcome = (
        VerificationStageOutcome.PASSED
        if disposition is VerificationDisposition.PASSED
        else VerificationStageOutcome.CHALLENGED
    )
    stage_records = []
    for stage_name in stage_kinds:
        stage_kind = validate_verification_stage_kind(stage_name)
        if stage_outcome is VerificationStageOutcome.PASSED:
            stage_records.append(
                verification_stage_record(
                    proposal_ref=proposal_ref,
                    stage=stage_kind,
                    outcome=stage_outcome,
                ),
            )
        else:
            finding = verification_finding(
                code=validate_verification_finding_code("verification.test.challenged"),
                message="challenged",
            )
            stage_records.append(
                verification_stage_record(
                    proposal_ref=proposal_ref,
                    stage=stage_kind,
                    outcome=stage_outcome,
                    challenge=verification_challenge(
                        proposal_ref=proposal_ref,
                        stage=stage_kind,
                        requirement_code=validate_verification_requirement_code(
                            f"verification.{stage_name}.agent_execution",
                        ),
                        finding=finding,
                    ),
                ),
            )
    verification = verification_result(
        proposal_ref=proposal_ref,
        disposition=disposition,
        stage_records=tuple(stage_records),
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
        human_review_pending=human_review_pending,
    )


def _human_review_pending_for_candidate(
    candidate,
) -> DecisionHumanReviewPending:
    request = decision_human_review_request(
        proposal_ref=candidate_decision_ref(candidate),
        reason_code=verification_challenged_human_review_reason(),
    )
    return request_decision_human_review(request)


def _hitl_historical_evidence() -> LegacyCriticHumanEscalationEvidence:
    return proven_retired_l2_human_escalation_evidence()


def _critic_verdict(
    *,
    passed: bool,
    action: CriticAction = CriticAction.CONTINUE,
    layers: tuple[CriticLayer, ...] | None = None,
) -> CriticVerdict:
    resolved_layers = layers or (CriticLayer.L0_DETERMINISTIC,)
    layer_verdicts = [
        LayerVerdict(
            layer=layer,
            passed=passed,
            score=1.0 if passed else 0.0,
            errors=[] if passed else ["layer failure"],
        )
        for layer in resolved_layers
    ]
    return CriticVerdict(
        scope=CriticScope.GRAPH_FINAL,
        passed=passed,
        layers=layer_verdicts,
        recommended_action=action,
        failure_reasons=[] if passed else ["structural failure"],
    )


def _structural_requirement() -> tuple[ParityCapabilityRequirement, ...]:
    return (
        ParityCapabilityRequirement(
            ParityVerificationCapability.STRUCTURAL,
            ParityCapabilityRequirementMode.CROSS_SYSTEM,
        ),
    )


def _semantic_requirement() -> tuple[ParityCapabilityRequirement, ...]:
    return (
        ParityCapabilityRequirement(
            ParityVerificationCapability.SEMANTIC,
            ParityCapabilityRequirementMode.CROSS_SYSTEM,
        ),
    )


def _evidence_superset_requirement() -> tuple[ParityCapabilityRequirement, ...]:
    return (
        ParityCapabilityRequirement(
            ParityVerificationCapability.EVIDENCE,
            ParityCapabilityRequirementMode.DECISION_SUPERSET,
        ),
    )


def _hitl_architectural_requirement() -> tuple[ParityCapabilityRequirement, ...]:
    return (
        ParityCapabilityRequirement(
            ParityVerificationCapability.HUMAN_HITL,
            ParityCapabilityRequirementMode.ARCHITECTURAL_MAPPING,
        ),
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
    assert parity.classification is DecisionCriticParityClassification.CAPABILITY_GAP
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
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=_decision_result(
            disposition=VerificationDisposition.CHALLENGED,
            host_action=DecisionFlowHostAction.PENDING_HUMAN,
        ),
        critic_verdict=_critic_verdict(passed=True),
        retired_legacy_human_evidence=_hitl_historical_evidence(),
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
        capability_requirements=_structural_requirement(),
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
        capability_requirements=_structural_requirement(),
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
        capability_requirements=_structural_requirement(),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityHostScope.UAEP_STEP in report.missing_scopes


def test_retirement_false_ready_regression_critic_only_semantic() -> None:
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
                stage_kinds=("structural",),
            ),
            critic_verdict=_critic_verdict(
                passed=True,
                layers=(CriticLayer.L1_SEMANTIC,),
            ),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_semantic_requirement(),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityVerificationCapability.SEMANTIC in report.missing_capabilities
    assert ParityVerificationCapability.SEMANTIC in report.critic_capabilities_exercised
    assert ParityVerificationCapability.SEMANTIC not in report.cross_system_capabilities_qualified


def test_cross_case_false_intersection_does_not_qualify() -> None:
    identity_a = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-a",
    )
    identity_b = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-b",
    )
    results = [
        compare_decision_critic_parity(
            identity=identity_a,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
                stage_kinds=("semantic",),
            ),
            critic_verdict=_critic_verdict(
                passed=True,
                layers=(CriticLayer.L0_DETERMINISTIC,),
            ),
        ),
        compare_decision_critic_parity(
            identity=identity_b,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
                stage_kinds=("structural",),
            ),
            critic_verdict=_critic_verdict(
                passed=True,
                layers=(CriticLayer.L1_SEMANTIC,),
            ),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=(
            ParityCapabilityRequirement(
                ParityVerificationCapability.SEMANTIC,
                ParityCapabilityRequirementMode.CROSS_SYSTEM,
            ),
            ParityCapabilityRequirement(
                ParityVerificationCapability.STRUCTURAL,
                ParityCapabilityRequirementMode.CROSS_SYSTEM,
            ),
        ),
    )
    assert ParityVerificationCapability.SEMANTIC not in report.cross_system_capabilities_qualified
    assert ParityVerificationCapability.STRUCTURAL not in report.cross_system_capabilities_qualified
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE


def test_valid_same_case_semantic_parity_qualifies() -> None:
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
                stage_kinds=("semantic",),
            ),
            critic_verdict=_critic_verdict(
                passed=True,
                layers=(CriticLayer.L1_SEMANTIC,),
            ),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_semantic_requirement(),
    )
    assert ParityVerificationCapability.SEMANTIC in report.cross_system_capabilities_qualified


def test_same_outcome_capability_gap_not_match() -> None:
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
            stage_kinds=("structural",),
        ),
        critic_verdict=_critic_verdict(
            passed=True,
            layers=(CriticLayer.L0_DETERMINISTIC, CriticLayer.L1_SEMANTIC),
        ),
    )
    assert parity.outcome_match is True
    assert parity.capability_match is False
    assert parity.classification is DecisionCriticParityClassification.CAPABILITY_GAP
    assert CRITIC_CAPABILITY_NOT_EXERCISED_BY_DECISION in {item.code for item in parity.differences}


def test_decision_superset_evidence_qualifies_without_critic() -> None:
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
                stage_kinds=("structural", "evidence"),
            ),
            critic_verdict=_critic_verdict(passed=True),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_evidence_superset_requirement(),
    )
    assert ParityVerificationCapability.EVIDENCE in report.decision_superset_capabilities_qualified
    assert ParityVerificationCapability.EVIDENCE not in report.missing_capabilities


def test_decision_superset_missing_evidence_is_insufficient() -> None:
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
                stage_kinds=("structural",),
            ),
            critic_verdict=_critic_verdict(passed=True),
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_evidence_superset_requirement(),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityVerificationCapability.EVIDENCE in report.missing_capabilities


def test_architectural_mapping_hitl_not_qualified_on_critic_l2_only() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    decision = _decision_result(
        disposition=VerificationDisposition.CHALLENGED,
        host_action=DecisionFlowHostAction.CONTINUE,
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=decision,
        critic_verdict=_critic_verdict(passed=True),
        retired_legacy_human_evidence=_hitl_historical_evidence(),
    )
    difference_codes = {item.code for item in parity.differences}
    assert LEGACY_L2_NOT_DECISION_VERIFICATION in difference_codes
    assert LEGACY_HITL_IS_DECISION_HUMAN_REVIEW not in difference_codes
    report = evaluate_critic_retirement_readiness(
        [parity],
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_hitl_architectural_requirement(),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityVerificationCapability.HUMAN_HITL not in report.architectural_mappings_qualified
    assert ParityVerificationCapability.HUMAN_HITL in report.missing_capabilities


def test_architectural_mapping_hitl_qualifies() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    decision = _decision_result(
        disposition=VerificationDisposition.CHALLENGED,
        host_action=DecisionFlowHostAction.PENDING_HUMAN,
    )
    decision_with_pending = replace(
        decision,
        human_review_pending=_human_review_pending_for_candidate(decision.candidate),
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=decision_with_pending,
            critic_verdict=_critic_verdict(passed=True),
            retired_legacy_human_evidence=_hitl_historical_evidence(),
        ),
    ]
    difference_codes = {item.code for item in results[0].differences}
    assert LEGACY_HITL_IS_DECISION_HUMAN_REVIEW in difference_codes
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_hitl_architectural_requirement(),
    )
    assert ParityVerificationCapability.HUMAN_HITL in report.architectural_mappings_qualified


def test_architectural_mapping_l2_difference_non_blocking_without_qualification() -> None:
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
            host_action=DecisionFlowHostAction.PENDING_HUMAN,
        ),
        critic_verdict=_critic_verdict(passed=True),
        retired_legacy_human_evidence=_hitl_historical_evidence(),
    )
    assert parity.classification is DecisionCriticParityClassification.EXPECTED_DIFFERENCE
    assert LEGACY_L2_NOT_DECISION_VERIFICATION in {item.code for item in parity.differences}
    assert not parity.retirement_blocking
    report = evaluate_critic_retirement_readiness(
        [parity],
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_hitl_architectural_requirement(),
    )
    assert ParityVerificationCapability.HUMAN_HITL not in report.architectural_mappings_qualified


def test_shadow_unavailable_does_not_qualify_hitl_mapping() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    decision = _decision_result(
        disposition=VerificationDisposition.CHALLENGED,
        host_action=DecisionFlowHostAction.PENDING_HUMAN,
    )
    decision_with_pending = replace(
        decision,
        human_review_pending=_human_review_pending_for_candidate(decision.candidate),
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=decision_with_pending,
            shadow_unavailable=True,
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_hitl_architectural_requirement(),
    )
    assert ParityVerificationCapability.HUMAN_HITL not in report.architectural_mappings_qualified
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE


def test_shadow_error_does_not_qualify_hitl_mapping() -> None:
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    decision = _decision_result(
        disposition=VerificationDisposition.CHALLENGED,
        host_action=DecisionFlowHostAction.PENDING_HUMAN,
    )
    decision_with_pending = replace(
        decision,
        human_review_pending=_human_review_pending_for_candidate(decision.candidate),
    )
    results = [
        compare_decision_critic_parity(
            identity=identity,
            decision_result=decision_with_pending,
            shadow_error="hitl shadow failed",
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_hitl_architectural_requirement(),
    )
    assert ParityVerificationCapability.HUMAN_HITL not in report.architectural_mappings_qualified
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE


def test_shadow_unavailable_semantic_without_alternate_is_insufficient() -> None:
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
                stage_kinds=("semantic",),
            ),
            shadow_unavailable=True,
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_semantic_requirement(),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert report.shadow_unavailable_count == 1


def test_shadow_unavailable_with_valid_semantic_pair_still_qualified() -> None:
    identity_a = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-a",
    )
    identity_b = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-b",
    )
    results = [
        compare_decision_critic_parity(
            identity=identity_a,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
                stage_kinds=("semantic",),
            ),
            critic_verdict=_critic_verdict(
                passed=True,
                layers=(CriticLayer.L1_SEMANTIC,),
            ),
        ),
        compare_decision_critic_parity(
            identity=identity_b,
            decision_result=_decision_result(
                disposition=VerificationDisposition.PASSED,
                host_action=DecisionFlowHostAction.CONTINUE,
                stage_kinds=("semantic",),
            ),
            shadow_unavailable=True,
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_semantic_requirement(),
    )
    assert ParityVerificationCapability.SEMANTIC in report.cross_system_capabilities_qualified
    assert ParityVerificationCapability.SEMANTIC not in report.missing_capabilities


def test_shadow_error_only_evidence_is_not_ready() -> None:
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
                stage_kinds=("semantic",),
            ),
            shadow_error="semantic shadow failed",
        ),
    ]
    report = evaluate_critic_retirement_readiness(
        results,
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_semantic_requirement(),
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert report.shadow_error_count == 1


def test_current_default_matrix_remains_insufficient_evidence() -> None:
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
        capability_requirements=DEFAULT_CRITIC_RETIREMENT_CAPABILITY_REQUIREMENTS,
    )
    assert report.readiness is CriticRetirementReadiness.INSUFFICIENT_EVIDENCE
    assert ParityVerificationCapability.STRUCTURAL in report.cross_system_capabilities_qualified
    assert ParityVerificationCapability.DETERMINISTIC_GUARDRAIL in report.missing_capabilities
    assert ParityVerificationCapability.SEMANTIC in report.missing_capabilities
    assert ParityVerificationCapability.TRAJECTORY in report.missing_capabilities
    assert ParityVerificationCapability.EVIDENCE in report.missing_capabilities
    assert ParityVerificationCapability.DOMAIN in report.missing_capabilities
    assert ParityVerificationCapability.HUMAN_HITL in report.missing_capabilities


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
    assert metrics.shadow_unavailable == 1


def test_ds_mig_03_hitl_qualified_without_live_l2_gateway() -> None:
    critic_root = Path("intergrax/runtime/critic")
    assert not (critic_root / "l2_gateway.py").is_file()
    identity = build_parity_identity(
        flow_scope=DecisionFlowScope.GRAPH_FINAL,
        task_id="task-1",
        run_id="run-1",
        attempt_id="attempt-1",
        tenant_id="tenant-1",
        agent_id="agent-1",
        subject="graph-1",
    )
    decision = _decision_result(
        disposition=VerificationDisposition.CHALLENGED,
        host_action=DecisionFlowHostAction.PENDING_HUMAN,
    )
    decision_with_pending = replace(
        decision,
        human_review_pending=_human_review_pending_for_candidate(decision.candidate),
    )
    parity = compare_decision_critic_parity(
        identity=identity,
        decision_result=decision_with_pending,
        critic_verdict=_critic_verdict(passed=True),
        retired_legacy_human_evidence=_hitl_historical_evidence(),
    )
    assert LEGACY_HITL_IS_DECISION_HUMAN_REVIEW in {item.code for item in parity.differences}
    report = evaluate_critic_retirement_readiness(
        [parity],
        required_scopes=frozenset({ParityHostScope.GRAPH_FINAL}),
        capability_requirements=_hitl_architectural_requirement(),
    )
    assert ParityVerificationCapability.HUMAN_HITL in report.architectural_mappings_qualified


def test_forbidden_audit_migration_modules() -> None:
    for path in _MODULE_PATHS:
        source = path.read_text(encoding="utf-8")
        for fragment in _FORBIDDEN_FRAGMENTS:
            assert fragment not in source
