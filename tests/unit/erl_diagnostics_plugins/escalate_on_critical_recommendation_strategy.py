# © Artur Czarnecki. All rights reserved.

"""External recommendation plugin — CRITICAL → ESCALATE_TO_OPERATOR (ERL-DIAG-001D acceptance)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics import (
    ExternalEffectReliabilityDiagnosticClassificationContext,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilityOperatorRecommendationKind,
    ExternalEffectReliabilityRecommendationDecision,
    ExternalEffectReliabilitySeverityDecision,
    ReliabilityDiagnosticRecommendationStrategyId,
    ReliabilityDiagnosticRecommendationStrategyVersion,
)

_PLUGIN_RECOMMENDATION_STRATEGY_ID = ReliabilityDiagnosticRecommendationStrategyId(
    "test.external.erl_diagnostics_plugins.escalate_on_critical.v1",
)
_PLUGIN_RECOMMENDATION_STRATEGY_VERSION = ReliabilityDiagnosticRecommendationStrategyVersion("1")


class EscalateOnCriticalRecommendationStrategy:
    @property
    def strategy_id(self) -> ReliabilityDiagnosticRecommendationStrategyId:
        return _PLUGIN_RECOMMENDATION_STRATEGY_ID

    @property
    def strategy_version(self) -> ReliabilityDiagnosticRecommendationStrategyVersion:
        return _PLUGIN_RECOMMENDATION_STRATEGY_VERSION

    def recommend(
        self,
        context: ExternalEffectReliabilityDiagnosticClassificationContext,
        severity: ExternalEffectReliabilitySeverityDecision,
    ) -> ExternalEffectReliabilityRecommendationDecision:
        if severity.severity is ExternalEffectReliabilityDiagnosticSeverity.CRITICAL:
            kind = ExternalEffectReliabilityOperatorRecommendationKind.ESCALATE_TO_OPERATOR
            reason_code = "plugin_critical_escalate"
            explanation = "External test plugin recommends operator escalation for critical severity."
        else:
            kind = ExternalEffectReliabilityOperatorRecommendationKind.OBSERVE
            reason_code = "plugin_observe"
            explanation = "External test plugin recommends observation for non-critical severity."
        return ExternalEffectReliabilityRecommendationDecision(
            recommendation_kind=kind,
            strategy_id=self.strategy_id,
            strategy_version=self.strategy_version,
            reason_code=reason_code,
            safe_explanation=explanation,
        )


__all__ = ["EscalateOnCriticalRecommendationStrategy"]
