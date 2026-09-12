# © Artur Czarnecki. All rights reserved.

"""Shared orchestration helpers for evolution intelligence plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionAnalysisFinding,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
    EvolutionIntelligenceInsight,
    EvolutionIntelligenceRecommendation,
    EvolutionMetricSnapshot,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.protocol import (
    EvolutionAnalyzerProvider,
    EvolutionInsightProvider,
    EvolutionMetricProvider,
    EvolutionRecommendationProvider,
)


def _collect_data_source_refs(
    findings: tuple[EvolutionAnalysisFinding, ...],
    metrics: tuple[EvolutionMetricSnapshot, ...],
    insights: tuple[EvolutionIntelligenceInsight, ...],
) -> tuple[EvolutionIntelligenceDataSourceRef, ...]:
    refs: list[EvolutionIntelligenceDataSourceRef] = []
    for item in findings:
        refs.extend(item.data_source_refs)
    for item in metrics:
        refs.extend(item.data_source_refs)
    for item in insights:
        refs.extend(item.evidence_source_refs)
    return tuple(dict.fromkeys(refs))


def run_evolution_intelligence_pipeline(
    context: EvolutionIntelligenceContext,
    *,
    analyzer_providers: tuple[EvolutionAnalyzerProvider, ...],
    metric_providers: tuple[EvolutionMetricProvider, ...],
    insight_providers: tuple[EvolutionInsightProvider, ...],
    recommendation_providers: tuple[EvolutionRecommendationProvider, ...],
) -> tuple[
    tuple[EvolutionAnalysisFinding, ...],
    tuple[EvolutionMetricSnapshot, ...],
    tuple[EvolutionIntelligenceInsight, ...],
    tuple[EvolutionIntelligenceRecommendation, ...],
    tuple[EvolutionIntelligenceDataSourceRef, ...],
]:
    findings = tuple(
        finding
        for provider in analyzer_providers
        for finding in provider.analyze(context)
    )
    metrics = tuple(
        metric
        for provider in metric_providers
        for metric in provider.collect(context, findings)
    )
    insights = tuple(
        insight
        for provider in insight_providers
        for insight in provider.derive(context, findings, metrics)
    )
    recommendations = tuple(
        recommendation
        for provider in recommendation_providers
        for recommendation in provider.recommend(context, insights)
    )
    data_refs = _collect_data_source_refs(findings, metrics, insights)
    return findings, metrics, insights, recommendations, data_refs


__all__ = [
    "run_evolution_intelligence_pipeline",
]
