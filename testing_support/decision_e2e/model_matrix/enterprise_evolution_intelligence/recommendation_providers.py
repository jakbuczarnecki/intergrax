# © Artur Czarnecki. All rights reserved.

"""Evolution recommendation provider plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionIntelligenceContext,
    EvolutionIntelligenceInsight,
    EvolutionIntelligenceRecommendation,
    EvolutionRecommendationKind,
)

_DEFAULT_RECOMMENDATION_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionRecommendationProvider:
    @property
    def provider_id(self) -> str:
        return "default_evolution_recommendation"

    @property
    def provider_version(self) -> str:
        return _DEFAULT_RECOMMENDATION_PROVIDER_VERSION

    def recommend(
        self,
        context: EvolutionIntelligenceContext,
        insights: tuple[EvolutionIntelligenceInsight, ...],
    ) -> tuple[EvolutionIntelligenceRecommendation, ...]:
        if not insights:
            return (
                EvolutionIntelligenceRecommendation(
                    recommendation_id=f"rec:collect:{context.adaptation_id}",
                    kind=EvolutionRecommendationKind.COLLECT_MORE_EVIDENCE,
                    summary="Collect more operational evidence before concluding.",
                    recommendation_provider_id=self.provider_id,
                    recommendation_provider_version=self.provider_version,
                    linked_insight_ids=(),
                    rationale="Insufficient insights from analyzers.",
                ),
            )
        primary = insights[0]
        kind = EvolutionRecommendationKind.CONTINUE_OBSERVATION
        summary = "Continue observation of adaptation outcomes."
        rationale = "Signals are stable; extended observation is appropriate."
        if "elevated" in primary.observation or "uncertain" in primary.observation:
            kind = EvolutionRecommendationKind.REVIEW_ADAPTATION
            summary = "Review adaptation effectiveness and cost trade-offs."
            rationale = "Analysis indicates elevated cost or uncertain effectiveness."
        return (
            EvolutionIntelligenceRecommendation(
                recommendation_id=f"rec:maintenance:{context.adaptation_id}",
                kind=kind,
                summary=summary,
                recommendation_provider_id=self.provider_id,
                recommendation_provider_version=self.provider_version,
                linked_insight_ids=tuple(item.insight_id for item in insights),
                rationale=rationale,
            ),
        )


def default_recommendation_providers() -> tuple[
    DefaultEvolutionRecommendationProvider,
]:
    return (DefaultEvolutionRecommendationProvider(),)


__all__ = [
    "DefaultEvolutionRecommendationProvider",
    "default_recommendation_providers",
]
