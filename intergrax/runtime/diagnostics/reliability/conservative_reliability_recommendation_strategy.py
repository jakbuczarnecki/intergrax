# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Conservative default ERL reliability recommendation strategy (ERL-DIAG-001D)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics.classification import (
    RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_ID,
    RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_VERSION,
    ExternalEffectReliabilityDiagnosticClassificationContext,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilityOperatorRecommendationKind,
    ExternalEffectReliabilityRecommendationDecision,
    ExternalEffectReliabilitySeverityDecision,
    ReliabilityDiagnosticRecommendationStrategyId,
    ReliabilityDiagnosticRecommendationStrategyVersion,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    ExternalEffectReliabilitySignalKind,
)


class ConservativeReliabilityRecommendationStrategy:
    """Advisory default mapping from classified severity and reliability facts."""

    @property
    def strategy_id(self) -> ReliabilityDiagnosticRecommendationStrategyId:
        return RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_ID

    @property
    def strategy_version(self) -> ReliabilityDiagnosticRecommendationStrategyVersion:
        return RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_VERSION

    def recommend(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
        severity: ExternalEffectReliabilitySeverityDecision,
    ) -> ExternalEffectReliabilityRecommendationDecision:
        if type(context) is not ExternalEffectReliabilityDiagnosticClassificationContext:
            raise TypeError("context must be ExternalEffectReliabilityDiagnosticClassificationContext")
        if type(severity) is not ExternalEffectReliabilitySeverityDecision:
            raise TypeError("severity must be ExternalEffectReliabilitySeverityDecision")
        kind, reason_code, explanation = _conservative_recommendation(
            context.signal_kind,
            severity.severity,
        )
        evidence_refs = _bounded_evidence_refs(context)
        return ExternalEffectReliabilityRecommendationDecision(
            recommendation_kind=kind,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            reason_code=reason_code,
            safe_explanation=explanation,
            evidence_refs=evidence_refs,
        )


def _conservative_recommendation(
    signal_kind: ExternalEffectReliabilitySignalKind,
    severity: ExternalEffectReliabilityDiagnosticSeverity,
) -> tuple[ExternalEffectReliabilityOperatorRecommendationKind, str, str]:
    if severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.ESCALATE_TO_OPERATOR,
            "severity_critical",
            "Escalate to an operator for review based on classified severity.",
        )
    if signal_kind is ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.WAIT_FOR_TRUTH,
            "truth_unavailable",
            "External truth is unavailable; wait or monitor before further automated handling.",
        )
    if signal_kind is ExternalEffectReliabilitySignalKind.EVIDENCE_INSUFFICIENT:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.REVIEW_EVIDENCE,
            "evidence_insufficient",
            "Review available evidence before further automated handling.",
        )
    if signal_kind is ExternalEffectReliabilitySignalKind.AUTOMATION_SAFETY_LIMIT:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.MANUAL_INVESTIGATION,
            "automation_safety_limit",
            "Automation safety limits apply; perform manual investigation.",
        )
    if signal_kind is ExternalEffectReliabilitySignalKind.GOVERNANCE_POSTURE:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.REQUEST_APPROVAL,
            "governance_posture",
            "Governance posture requires operator review; request approval if applicable.",
        )
    if signal_kind is ExternalEffectReliabilitySignalKind.TRUTH_ESTABLISHED:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.NO_OPERATOR_ACTION_REQUIRED,
            "truth_established",
            "Truth is established; no operator action required from this signal alone.",
        )
    if signal_kind is ExternalEffectReliabilitySignalKind.RECOVERY_POSTURE:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE,
            "recovery_posture",
            "Observe recovery posture and linked evidence.",
        )
    if severity is ExternalEffectReliabilityDiagnosticSeverity.ERROR:
        return (
            ExternalEffectReliabilityOperatorRecommendationKind.MANUAL_INVESTIGATION,
            "severity_error",
            "Review available evidence before further automated handling.",
        )
    return (
        ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE,
        "default_observe",
        "Observe the reliability signal and linked evidence.",
    )


def _bounded_evidence_refs(
    context: ExternalEffectReliabilityDiagnosticClassificationContext,
) -> tuple[str, ...]:
    refs: list[str] = []
    artifact_refs = context.artifact_refs
    for candidate in (
        artifact_refs.evidence_ref,
        artifact_refs.reconciliation_ref,
        artifact_refs.resolution_ref,
    ):
        if candidate is not None and candidate.strip():
            refs.append(candidate.strip())
    return tuple(refs)


__all__ = ["ConservativeReliabilityRecommendationStrategy"]
