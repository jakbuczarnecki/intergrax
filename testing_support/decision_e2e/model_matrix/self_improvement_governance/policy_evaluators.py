# © Artur Czarnecki. All rights reserved.

"""Default self-improvement policy plugins (DS-E2E-15J-L12)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.autonomous_decision_evolution.contracts import (
    EvaluationCriterionKind,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    EvolutionRiskImpactLevel,
    SelfImprovementGovernanceReason,
    SelfImprovementGovernanceRequest,
    SelfImprovementGovernanceStatus,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.protocol import (
    SelfImprovementPolicyResult,
)

_EVIDENCE_POLICY_ID = "evidence_completeness"
_EVIDENCE_POLICY_VERSION = "1"
_RISK_POLICY_ID = "evolution_risk"
_RISK_POLICY_VERSION = "1"
_QUALITY_POLICY_ID = "quality"
_QUALITY_POLICY_VERSION = "1"
_SAFETY_POLICY_ID = "safety"
_SAFETY_POLICY_VERSION = "1"


def _reason(
    code: str,
    summary: str,
    policy_id: str,
) -> SelfImprovementGovernanceReason:
    return SelfImprovementGovernanceReason(
        reason_code=code,
        summary=summary,
        policy_id=policy_id,
    )


def _approved(
    policy_id: str, version: str, summary: str
) -> SelfImprovementPolicyResult:
    return SelfImprovementPolicyResult(
        policy_id=policy_id,
        policy_version=version,
        status_contribution=SelfImprovementGovernanceStatus.APPROVED,
        reasons=(_reason("policy_allow", summary, policy_id),),
        required_actions=(),
    )


@dataclass(frozen=True, slots=True)
class EvidenceCompletenessPolicy:
    """Requires evaluation findings before evolution may proceed."""

    @property
    def policy_id(self) -> str:
        return _EVIDENCE_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _EVIDENCE_POLICY_VERSION

    def evaluate(
        self, request: SelfImprovementGovernanceRequest
    ) -> SelfImprovementPolicyResult:
        if not request.evaluation_findings:
            return SelfImprovementPolicyResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                status_contribution=SelfImprovementGovernanceStatus.REQUIRES_MORE_EVIDENCE,
                reasons=(
                    _reason(
                        "missing_evaluation_evidence",
                        "Evolution proposal lacks required evaluation findings.",
                        self.policy_id,
                    ),
                ),
                required_actions=("collect_experiment_evaluation_evidence",),
            )
        if request.experiment_result.experiment_id == "":
            return SelfImprovementPolicyResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                status_contribution=SelfImprovementGovernanceStatus.REQUIRES_MORE_EVIDENCE,
                reasons=(
                    _reason(
                        "missing_experiment_reference",
                        "Experiment result reference is incomplete.",
                        self.policy_id,
                    ),
                ),
                required_actions=("complete_controlled_experiment",),
            )
        return _approved(
            self.policy_id,
            self.policy_version,
            "Evaluation evidence present for governance review.",
        )


@dataclass(frozen=True, slots=True)
class EvolutionRiskPolicy:
    """Maps assessed risk findings to governance status — no standalone blocking."""

    @property
    def policy_id(self) -> str:
        return _RISK_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _RISK_POLICY_VERSION

    def evaluate(
        self, request: SelfImprovementGovernanceRequest
    ) -> SelfImprovementPolicyResult:
        findings = request.risk_context.findings
        if any(
            item.impact_level is EvolutionRiskImpactLevel.CRITICAL for item in findings
        ):
            return SelfImprovementPolicyResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                status_contribution=SelfImprovementGovernanceStatus.REJECTED,
                reasons=(
                    _reason(
                        "critical_evolution_risk",
                        "Critical risk findings prohibit accepting this evolution.",
                        self.policy_id,
                    ),
                ),
                required_actions=("escalate_to_risk_committee",),
            )
        if any(item.impact_level is EvolutionRiskImpactLevel.HIGH for item in findings):
            return SelfImprovementPolicyResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                status_contribution=SelfImprovementGovernanceStatus.REQUIRES_REVIEW,
                reasons=(
                    _reason(
                        "high_evolution_risk",
                        "High-impact evolution requires human governance review.",
                        self.policy_id,
                    ),
                ),
                required_actions=("schedule_human_governance_review",),
            )
        return _approved(
            self.policy_id,
            self.policy_version,
            "No elevated evolution risk findings.",
        )


@dataclass(frozen=True, slots=True)
class QualityEvolutionPolicy:
    """Quality dimension must support the proposal when quality findings exist."""

    @property
    def policy_id(self) -> str:
        return _QUALITY_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _QUALITY_POLICY_VERSION

    def evaluate(
        self, request: SelfImprovementGovernanceRequest
    ) -> SelfImprovementPolicyResult:
        quality_findings = tuple(
            item
            for item in request.evaluation_findings
            if item.criterion_kind is EvaluationCriterionKind.QUALITY
        )
        if not quality_findings:
            return _approved(
                self.policy_id,
                self.policy_version,
                "No quality-specific findings to assess.",
            )
        if all(item.supports_proposal for item in quality_findings):
            return _approved(
                self.policy_id,
                self.policy_version,
                "Quality evaluation supports the evolution proposal.",
            )
        return SelfImprovementPolicyResult(
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            status_contribution=SelfImprovementGovernanceStatus.REQUIRES_REVIEW,
            reasons=(
                _reason(
                    "quality_not_supported",
                    "Quality evaluation does not support the proposed evolution.",
                    self.policy_id,
                ),
            ),
            required_actions=("revise_proposal_or_gather_quality_evidence",),
        )


@dataclass(frozen=True, slots=True)
class SafetyEvolutionPolicy:
    """Safety findings that oppose the proposal block acceptance."""

    @property
    def policy_id(self) -> str:
        return _SAFETY_POLICY_ID

    @property
    def policy_version(self) -> str:
        return _SAFETY_POLICY_VERSION

    def evaluate(
        self, request: SelfImprovementGovernanceRequest
    ) -> SelfImprovementPolicyResult:
        safety_findings = tuple(
            item
            for item in request.evaluation_findings
            if item.criterion_kind is EvaluationCriterionKind.SAFETY
        )
        if any(not item.supports_proposal for item in safety_findings):
            return SelfImprovementPolicyResult(
                policy_id=self.policy_id,
                policy_version=self.policy_version,
                status_contribution=SelfImprovementGovernanceStatus.REJECTED,
                reasons=(
                    _reason(
                        "safety_opposition",
                        "Safety evaluation opposes the evolution proposal.",
                        self.policy_id,
                    ),
                ),
                required_actions=("halt_evolution_pending_safety_remediation",),
            )
        return _approved(
            self.policy_id,
            self.policy_version,
            "Safety evaluation does not oppose the proposal.",
        )


def default_policy_evaluators() -> tuple[
    EvidenceCompletenessPolicy
    | EvolutionRiskPolicy
    | QualityEvolutionPolicy
    | SafetyEvolutionPolicy,
    ...,
]:
    return (
        EvidenceCompletenessPolicy(),
        EvolutionRiskPolicy(),
        QualityEvolutionPolicy(),
        SafetyEvolutionPolicy(),
    )


__all__ = [
    "EvidenceCompletenessPolicy",
    "EvolutionRiskPolicy",
    "QualityEvolutionPolicy",
    "SafetyEvolutionPolicy",
    "default_policy_evaluators",
]
