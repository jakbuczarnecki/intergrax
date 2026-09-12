# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    DecisionEvolutionProposal,
    EvaluationCriterionKind,
    EvolutionEvaluationFinding,
    EvolutionExperimentSpec,
    EvolutionRiskInformation,
    EvolutionTargetArea,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance import (
    EvolutionRiskContext,
    EvolutionRiskPolicy,
    RecordedSelfImprovementApprovalProvider,
    SelfImprovementGovernanceEngine,
    SelfImprovementGovernanceRequest,
    SelfImprovementGovernanceStatus,
    default_policy_evaluators,
    default_risk_evaluators,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.audit_providers import (
    default_audit_provider,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.policy_evaluators import (
    EvidenceCompletenessPolicy,
    SafetyEvolutionPolicy,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.protocol import (
    SelfImprovementPolicyEvaluator,
)


def _stamp() -> datetime:
    return datetime(2026, 9, 12, 16, 0, 0, tzinfo=UTC)


def _proposal(
    *,
    proposal_id: str = "prop-1",
    risk_summary: str = "Low operational impact with documented mitigations.",
) -> DecisionEvolutionProposal:
    return DecisionEvolutionProposal(
        proposal_id=proposal_id,
        generator_id="test_gen",
        generator_version="1",
        source_insight_ids=("ins-1",),
        target_area=EvolutionTargetArea.MODEL_SELECTION,
        proposed_change="Adjust model routing heuristic for documents.",
        expected_benefit="Reduced approval friction.",
        risk_information=EvolutionRiskInformation(
            risk_summary=risk_summary,
            mitigation_notes="Shadow evaluation only.",
        ),
        created_at=_stamp(),
    )


def _experiment() -> EvolutionExperimentSpec:
    return EvolutionExperimentSpec(
        experiment_id="exp-1",
        proposal_id="prop-1",
        experiment_provider_id="exp_provider",
        experiment_provider_version="1",
        variants=(),
        evaluation_criteria=(),
        experiment_summary="Controlled shadow comparison.",
    )


def _quality_finding(supports: bool = True) -> EvolutionEvaluationFinding:
    return EvolutionEvaluationFinding(
        evaluation_id="eval-quality-1",
        experiment_id="exp-1",
        evaluator_id="quality_eval",
        evaluator_version="1",
        criterion_kind=EvaluationCriterionKind.QUALITY,
        finding_summary="Quality dimension assessed.",
        supports_proposal=supports,
    )


def _safety_finding(supports: bool = True) -> EvolutionEvaluationFinding:
    return EvolutionEvaluationFinding(
        evaluation_id="eval-safety-1",
        experiment_id="exp-1",
        evaluator_id="safety_eval",
        evaluator_version="1",
        criterion_kind=EvaluationCriterionKind.SAFETY,
        finding_summary="Safety dimension assessed.",
        supports_proposal=supports,
    )


def _engine(
    *,
    approval: RecordedSelfImprovementApprovalProvider | None = None,
    policies: tuple[SelfImprovementPolicyEvaluator, ...] | None = None,
) -> SelfImprovementGovernanceEngine:
    return SelfImprovementGovernanceEngine(
        policy_evaluators=policies or default_policy_evaluators(),
        risk_evaluators=default_risk_evaluators(),
        approval_provider=approval
        or RecordedSelfImprovementApprovalProvider(
            outcome_status=SelfImprovementGovernanceStatus.APPROVED
        ),
        audit_provider=default_audit_provider(),
    )


def _request(
    *,
    findings: tuple[EvolutionEvaluationFinding, ...] | None = None,
    risk_summary: str = "Low operational impact with documented mitigations.",
) -> SelfImprovementGovernanceRequest:
    proposal = _proposal(risk_summary=risk_summary)
    return SelfImprovementGovernanceRequest(
        evolution_proposal=proposal,
        experiment_result=_experiment(),
        evaluation_findings=(
            findings
            if findings is not None
            else (_quality_finding(), _safety_finding())
        ),
        risk_context=EvolutionRiskContext(
            proposal_id=proposal.proposal_id,
            findings=(),
        ),
    )


def test_governance_approves_complete_low_risk_proposal() -> None:
    decision = _engine().evaluate_governance(_request(), evaluated_at=_stamp())
    assert decision.status is SelfImprovementGovernanceStatus.APPROVED
    assert decision.audit_metadata.proposal_reference == "prop-1"
    assert decision.audit_metadata.evaluation_references == (
        "eval-quality-1",
        "eval-safety-1",
    )


def test_governance_requires_more_evidence_when_findings_missing() -> None:
    decision = _engine().evaluate_governance(
        _request(findings=()),
        evaluated_at=_stamp(),
    )
    assert decision.status is SelfImprovementGovernanceStatus.REQUIRES_MORE_EVIDENCE
    assert "collect_experiment_evaluation_evidence" in decision.required_actions


def test_governance_requires_review_for_high_impact_proposal() -> None:
    decision = _engine().evaluate_governance(
        _request(risk_summary="High impact change to production routing."),
        evaluated_at=_stamp(),
    )
    assert decision.status is SelfImprovementGovernanceStatus.REQUIRES_REVIEW


def test_policy_plugin_swap_without_engine_change() -> None:
    strict_policies = (
        EvidenceCompletenessPolicy(),
        EvolutionRiskPolicy(),
        SafetyEvolutionPolicy(),
    )
    engine = _engine(policies=strict_policies)
    decision = engine.evaluate_governance(
        _request(findings=(_safety_finding(supports=False),)),
        evaluated_at=_stamp(),
    )
    assert decision.status is SelfImprovementGovernanceStatus.REJECTED


def test_approval_provider_swap_without_core_change() -> None:
    engine = _engine(
        approval=RecordedSelfImprovementApprovalProvider(
            outcome_status=SelfImprovementGovernanceStatus.REQUIRES_REVIEW,
            provider_id="enterprise_workflow",
        )
    )
    decision = engine.evaluate_governance(_request(), evaluated_at=_stamp())
    assert decision.status is SelfImprovementGovernanceStatus.REQUIRES_REVIEW
    assert decision.audit_metadata.approval_provider_id == "enterprise_workflow"
