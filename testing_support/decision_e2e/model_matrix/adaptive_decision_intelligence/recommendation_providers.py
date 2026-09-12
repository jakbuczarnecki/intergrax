# © Artur Czarnecki. All rights reserved.

"""Built-in adaptive recommendation providers (extend via new classes)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDecisionIntelligenceContext,
    AdaptiveDecisionRecommendation,
    AdaptiveReasoningInsight,
    ConfidenceLevel,
)


class TechnicalRecommendationProvider:
    """Technical framing — does not execute routing or policy changes."""

    provider_id = "technical_recommendation"
    provider_version = "1"
    recommendation_kind = "technical"

    def recommend(
        self,
        reasoning_insights: tuple[AdaptiveReasoningInsight, ...],
        *,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveDecisionRecommendation, ...]:
        del context
        recommendations: list[AdaptiveDecisionRecommendation] = []
        for insight in reasoning_insights:
            narrative = _technical_narrative(insight)
            recommendations.append(
                AdaptiveDecisionRecommendation(
                    recommendation_id=f"{self.provider_id}:{insight.reasoning_insight_id}",
                    recommendation_provider_id=self.provider_id,
                    recommendation_provider_version=self.provider_version,
                    recommendation_kind=self.recommendation_kind,
                    narrative=narrative,
                    linked_reasoning_insight_ids=(insight.reasoning_insight_id,),
                    confidence=insight.confidence,
                    confidence_score=insight.confidence_score,
                )
            )
        return tuple(recommendations)


class BusinessRecommendationProvider:
    """Business framing — advisory for human acceptance only."""

    provider_id = "business_recommendation"
    provider_version = "1"
    recommendation_kind = "business"

    def recommend(
        self,
        reasoning_insights: tuple[AdaptiveReasoningInsight, ...],
        *,
        context: AdaptiveDecisionIntelligenceContext,
    ) -> tuple[AdaptiveDecisionRecommendation, ...]:
        del context
        if not reasoning_insights:
            return ()
        best = _highest_confidence_insight(reasoning_insights)
        return (
            AdaptiveDecisionRecommendation(
                recommendation_id=f"{self.provider_id}:{best.reasoning_insight_id}",
                recommendation_provider_id=self.provider_id,
                recommendation_provider_version=self.provider_version,
                recommendation_kind=self.recommendation_kind,
                narrative=(
                    "Consider an alternative model or additional human review — "
                    "this is a suggestion only; no automatic change is applied."
                ),
                linked_reasoning_insight_ids=tuple(
                    item.reasoning_insight_id for item in reasoning_insights
                ),
                confidence=best.confidence,
                confidence_score=best.confidence_score,
            ),
        )


def _technical_narrative(insight: AdaptiveReasoningInsight) -> str:
    if insight.reasoning_provider_id == "risk_reasoning":
        return (
            "Technical advisory: elevated historical risk — evaluate alternate "
            "models or safeguards before proceeding."
        )
    if insight.reasoning_provider_id == "quality_reasoning":
        return (
            "Technical advisory: leverage capability baseline evidence when "
            "choosing model parameters."
        )
    return (
        f"Technical advisory based on analysis: {insight.reasoning_summary} "
        "(human or upstream process decides)."
    )


def _highest_confidence_insight(
    insights: tuple[AdaptiveReasoningInsight, ...],
) -> AdaptiveReasoningInsight:
    order = (ConfidenceLevel.LOW, ConfidenceLevel.MEDIUM, ConfidenceLevel.HIGH)
    best = insights[0]
    for item in insights[1:]:
        if order.index(item.confidence) > order.index(best.confidence):
            best = item
    return best


def default_recommendation_providers() -> tuple[TechnicalRecommendationProvider,]:
    return (TechnicalRecommendationProvider(),)


__all__ = [
    "BusinessRecommendationProvider",
    "TechnicalRecommendationProvider",
    "default_recommendation_providers",
]
