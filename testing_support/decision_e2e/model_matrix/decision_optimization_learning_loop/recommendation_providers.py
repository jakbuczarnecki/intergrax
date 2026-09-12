# © Artur Czarnecki. All rights reserved.

"""Built-in recommendation providers (extend via new classes)."""

from __future__ import annotations

from datetime import datetime

from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    DecisionOptimizationContext,
    DecisionOptimizationSuggestion,
    OptimizationArea,
    OptimizationInsight,
)


class HumanReviewRecommendationProvider:
    """Produces review-oriented suggestions — no runtime or policy mutations."""

    provider_id = "human_review"
    provider_version = "1"

    def recommend(
        self,
        insights: tuple[OptimizationInsight, ...],
        *,
        context: DecisionOptimizationContext,
        generated_at: datetime,
    ) -> tuple[DecisionOptimizationSuggestion, ...]:
        del context
        suggestions: list[DecisionOptimizationSuggestion] = []
        for insight in insights:
            rationale = _rationale_for_area(insight)
            suggestions.append(
                DecisionOptimizationSuggestion(
                    suggestion_id=f"{self.provider_id}:{insight.insight_id}",
                    source_decisions=insight.source_decision_ids,
                    optimization_area=insight.optimization_area,
                    observation_references=insight.data_source_refs,
                    rationale=rationale,
                    confidence=insight.confidence,
                    generated_at=generated_at,
                    provider_id=self.provider_id,
                    provider_version=self.provider_version,
                    insight_ids=(insight.insight_id,),
                )
            )
        return tuple(suggestions)


def _rationale_for_area(insight: OptimizationInsight) -> str:
    if insight.optimization_area is OptimizationArea.GOVERNANCE_FRICTION:
        return (
            "Consider re-evaluating governance thresholds or alternative models "
            "that historically incur fewer blocks — human or upstream process decides."
        )
    if insight.optimization_area is OptimizationArea.OUTCOME_RELIABILITY:
        return (
            "Consider reviewing decision paths linked to failed or blocked outcomes "
            "for process improvements — no automatic execution change."
        )
    if insight.optimization_area is OptimizationArea.MODEL_CAPABILITY:
        return (
            "Consider reassessing model usage where capability profiles show "
            "limitation-heavy evidence — recommendation only."
        )
    return (
        f"Review optimization opportunity: {insight.narrative} "
        "(suggestion for human consideration only)."
    )


def default_recommendation_providers() -> tuple[HumanReviewRecommendationProvider,]:
    return (HumanReviewRecommendationProvider(),)


__all__ = [
    "HumanReviewRecommendationProvider",
    "default_recommendation_providers",
]
