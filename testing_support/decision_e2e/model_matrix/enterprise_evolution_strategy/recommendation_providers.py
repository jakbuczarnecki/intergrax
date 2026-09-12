# © Artur Czarnecki. All rights reserved.

"""Evolution strategy recommendation provider plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionFutureScenario,
    EvolutionScenarioImpactAssessment,
    EvolutionStrategicRecommendation,
    EvolutionStrategyContext,
    EvolutionStrategyRecommendationKind,
)

_DEFAULT_STRATEGY_RECOMMENDATION_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionStrategyRecommendationProvider:
    @property
    def provider_id(self) -> str:
        return "default_evolution_strategy_recommendation"

    @property
    def provider_version(self) -> str:
        return _DEFAULT_STRATEGY_RECOMMENDATION_VERSION

    def recommend(
        self,
        context: EvolutionStrategyContext,
        scenarios: tuple[EvolutionFutureScenario, ...],
        impact_assessments: tuple[EvolutionScenarioImpactAssessment, ...],
    ) -> tuple[EvolutionStrategicRecommendation, ...]:
        if not scenarios:
            return (
                EvolutionStrategicRecommendation(
                    recommendation_id=f"rec:collect:{context.scope_id}",
                    kind=EvolutionStrategyRecommendationKind.COLLECT_MORE_DATA,
                    summary="Collect more evolution history before strategic scenarios.",
                    recommendation_provider_id=self.provider_id,
                    recommendation_provider_version=self.provider_version,
                    linked_scenario_ids=(),
                    linked_assessment_ids=(),
                    rationale="No scenarios generated from analyzers.",
                ),
            )
        scenario_ids = tuple(item.scenario_id for item in scenarios)
        assessment_ids = tuple(item.assessment_id for item in impact_assessments)
        has_capability_finding = any(
            obs.trend_direction == "increasing"
            for obs in context.capability_observations
        )
        kind = EvolutionStrategyRecommendationKind.REVIEW
        summary = (
            "Review strategic directions with architecture and governance stakeholders."
        )
        rationale = "Scenarios and impacts are available for human strategic review."
        if has_capability_finding:
            kind = EvolutionStrategyRecommendationKind.EXPLORE
            summary = (
                "Explore greater use of specialist capabilities based on "
                "observed trends — architecture change remains human-led."
            )
            rationale = "Capability observations suggest a direction worth exploration."
        if not impact_assessments:
            kind = EvolutionStrategyRecommendationKind.COLLECT_MORE_DATA
            summary = "Collect impact evidence before strategic recommendations."
            rationale = "Impact assessments missing for generated scenarios."
        return (
            EvolutionStrategicRecommendation(
                recommendation_id=f"rec:strategy:{context.scope_id}",
                kind=kind,
                summary=summary,
                recommendation_provider_id=self.provider_id,
                recommendation_provider_version=self.provider_version,
                linked_scenario_ids=scenario_ids,
                linked_assessment_ids=assessment_ids,
                rationale=rationale,
            ),
        )


def default_strategy_recommendation_providers() -> tuple[
    DefaultEvolutionStrategyRecommendationProvider,
]:
    return (DefaultEvolutionStrategyRecommendationProvider(),)


__all__ = [
    "DefaultEvolutionStrategyRecommendationProvider",
    "default_strategy_recommendation_providers",
]
