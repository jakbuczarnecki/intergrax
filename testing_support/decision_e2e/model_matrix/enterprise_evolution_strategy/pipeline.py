# © Artur Czarnecki. All rights reserved.

"""Shared orchestration helpers for evolution strategy plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionFutureScenario,
    EvolutionScenarioImpactAssessment,
    EvolutionStrategicRecommendation,
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
    EvolutionStrategyDirectionFinding,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.protocol import (
    EvolutionImpactAnalyzerProvider,
    EvolutionScenarioProvider,
    EvolutionStrategyAnalyzerProvider,
    EvolutionStrategyRecommendationProvider,
)


def _collect_data_source_refs(
    findings: tuple[EvolutionStrategyDirectionFinding, ...],
    scenarios: tuple[EvolutionFutureScenario, ...],
    impact_assessments: tuple[EvolutionScenarioImpactAssessment, ...],
) -> tuple[EvolutionStrategyDataSourceRef, ...]:
    refs: list[EvolutionStrategyDataSourceRef] = []
    for item in findings:
        refs.extend(item.data_source_refs)
    for item in scenarios:
        refs.extend(item.data_source_refs)
    for item in impact_assessments:
        refs.extend(item.data_source_refs)
    return tuple(dict.fromkeys(refs))


def run_evolution_strategy_pipeline(
    context: EvolutionStrategyContext,
    *,
    strategy_analyzer_providers: tuple[EvolutionStrategyAnalyzerProvider, ...],
    scenario_providers: tuple[EvolutionScenarioProvider, ...],
    impact_analyzer_providers: tuple[EvolutionImpactAnalyzerProvider, ...],
    recommendation_providers: tuple[EvolutionStrategyRecommendationProvider, ...],
) -> tuple[
    tuple[EvolutionStrategyDirectionFinding, ...],
    tuple[EvolutionFutureScenario, ...],
    tuple[EvolutionScenarioImpactAssessment, ...],
    tuple[EvolutionStrategicRecommendation, ...],
    tuple[EvolutionStrategyDataSourceRef, ...],
]:
    findings = tuple(
        finding
        for provider in strategy_analyzer_providers
        for finding in provider.analyze(context)
    )
    scenarios = tuple(
        scenario
        for provider in scenario_providers
        for scenario in provider.build_scenarios(context, findings)
    )
    impact_assessments = tuple(
        assessment
        for provider in impact_analyzer_providers
        for assessment in provider.assess(context, scenarios)
    )
    recommendations = tuple(
        recommendation
        for provider in recommendation_providers
        for recommendation in provider.recommend(context, scenarios, impact_assessments)
    )
    data_refs = _collect_data_source_refs(findings, scenarios, impact_assessments)
    return findings, scenarios, impact_assessments, recommendations, data_refs


__all__ = [
    "run_evolution_strategy_pipeline",
]
