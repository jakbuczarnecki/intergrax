# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution intelligence orchestration via injected plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceResult,
    EvolutionIntelligenceRunStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.pipeline import (
    run_evolution_intelligence_pipeline,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.protocol import (
    EnterpriseEvolutionIntelligenceProvider,
    EvolutionAnalyzerProvider,
    EvolutionInsightProvider,
    EvolutionIntelligenceAuditProvider,
    EvolutionMetricProvider,
    EvolutionRecommendationProvider,
)


def _has_input_data(context: EvolutionIntelligenceContext) -> bool:
    return bool(
        context.operation_records
        or context.execution_results
        or context.health_observations
        or context.controlled_evolution is not None
    )


def _provider_ids(
    providers: tuple[
        EnterpriseEvolutionIntelligenceProvider
        | EvolutionAnalyzerProvider
        | EvolutionMetricProvider
        | EvolutionInsightProvider
        | EvolutionRecommendationProvider,
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_id for item in providers)


def _provider_versions(
    providers: tuple[
        EnterpriseEvolutionIntelligenceProvider
        | EvolutionAnalyzerProvider
        | EvolutionMetricProvider
        | EvolutionInsightProvider
        | EvolutionRecommendationProvider,
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_version for item in providers)


@dataclass(frozen=True, slots=True)
class EnterpriseEvolutionIntelligenceEngine:
    intelligence_providers: tuple[EnterpriseEvolutionIntelligenceProvider, ...]
    analyzer_providers: tuple[EvolutionAnalyzerProvider, ...]
    metric_providers: tuple[EvolutionMetricProvider, ...]
    insight_providers: tuple[EvolutionInsightProvider, ...]
    recommendation_providers: tuple[EvolutionRecommendationProvider, ...]
    audit_provider: EvolutionIntelligenceAuditProvider

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
        *,
        analyzed_at: datetime | None = None,
    ) -> EvolutionIntelligenceResult:
        stamp = analyzed_at or datetime.now(tz=UTC)
        empty_audit = self.audit_provider.build_audit(
            context,
            intelligence_provider_ids=_provider_ids(self.intelligence_providers),
            intelligence_provider_versions=_provider_versions(
                self.intelligence_providers
            ),
            analyzer_ids=_provider_ids(self.analyzer_providers),
            analyzer_versions=_provider_versions(self.analyzer_providers),
            metric_provider_ids=_provider_ids(self.metric_providers),
            metric_provider_versions=_provider_versions(self.metric_providers),
            insight_provider_ids=_provider_ids(self.insight_providers),
            insight_provider_versions=_provider_versions(self.insight_providers),
            recommendation_provider_ids=_provider_ids(self.recommendation_providers),
            recommendation_provider_versions=_provider_versions(
                self.recommendation_providers
            ),
            data_source_refs=(),
            analyzed_at=stamp,
            analysis_scope_summary="No evolution history supplied for analysis.",
        )
        if not _has_input_data(context):
            return EvolutionIntelligenceResult(
                intelligence_task_id=ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
                status=EvolutionIntelligenceRunStatus.INSUFFICIENT_INPUT,
                audit=empty_audit,
                analysis_findings=(),
                metric_snapshots=(),
                insights=(),
                recommendations=(),
            )

        findings, metrics, insights, recommendations, data_refs = (
            run_evolution_intelligence_pipeline(
                context,
                analyzer_providers=self.analyzer_providers,
                metric_providers=self.metric_providers,
                insight_providers=self.insight_providers,
                recommendation_providers=self.recommendation_providers,
            )
        )

        supplemental_findings = findings
        supplemental_metrics = metrics
        supplemental_insights = insights
        supplemental_recommendations = recommendations
        supplemental_refs = data_refs

        for provider in self.intelligence_providers:
            partial = provider.analyze(context, analyzed_at=stamp)
            supplemental_findings = (
                *supplemental_findings,
                *partial.analysis_findings,
            )
            supplemental_metrics = (*supplemental_metrics, *partial.metric_snapshots)
            supplemental_insights = (*supplemental_insights, *partial.insights)
            supplemental_recommendations = (
                *supplemental_recommendations,
                *partial.recommendations,
            )
            supplemental_refs = tuple(
                dict.fromkeys((*supplemental_refs, *partial.audit.data_source_refs))
            )

        scope_summary = (
            f"Analyzed adaptation {context.adaptation_id} v{context.version} "
            f"with {len(context.operation_records)} operations, "
            f"{len(context.execution_results)} executions, "
            f"{len(context.health_observations)} health observations."
        )
        audit = self.audit_provider.build_audit(
            context,
            intelligence_provider_ids=_provider_ids(self.intelligence_providers),
            intelligence_provider_versions=_provider_versions(
                self.intelligence_providers
            ),
            analyzer_ids=_provider_ids(self.analyzer_providers),
            analyzer_versions=_provider_versions(self.analyzer_providers),
            metric_provider_ids=_provider_ids(self.metric_providers),
            metric_provider_versions=_provider_versions(self.metric_providers),
            insight_provider_ids=_provider_ids(self.insight_providers),
            insight_provider_versions=_provider_versions(self.insight_providers),
            recommendation_provider_ids=_provider_ids(self.recommendation_providers),
            recommendation_provider_versions=_provider_versions(
                self.recommendation_providers
            ),
            data_source_refs=supplemental_refs,
            analyzed_at=stamp,
            analysis_scope_summary=scope_summary,
        )
        status = (
            EvolutionIntelligenceRunStatus.COMPLETE
            if supplemental_findings
            or supplemental_insights
            or supplemental_recommendations
            else EvolutionIntelligenceRunStatus.INSUFFICIENT_INPUT
        )
        return EvolutionIntelligenceResult(
            intelligence_task_id=ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
            status=status,
            audit=audit,
            analysis_findings=supplemental_findings,
            metric_snapshots=supplemental_metrics,
            insights=supplemental_insights,
            recommendations=supplemental_recommendations,
        )


def default_enterprise_evolution_intelligence_engine(
    *,
    intelligence_providers: tuple[EnterpriseEvolutionIntelligenceProvider, ...]
    | None = None,
    analyzer_providers: tuple[EvolutionAnalyzerProvider, ...] | None = None,
    metric_providers: tuple[EvolutionMetricProvider, ...] | None = None,
    insight_providers: tuple[EvolutionInsightProvider, ...] | None = None,
    recommendation_providers: tuple[EvolutionRecommendationProvider, ...] | None = None,
    audit_provider: EvolutionIntelligenceAuditProvider | None = None,
) -> EnterpriseEvolutionIntelligenceEngine:
    if audit_provider is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.audit_providers import (
            default_evolution_intelligence_audit_provider,
        )

        audit_provider = default_evolution_intelligence_audit_provider()
    if analyzer_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.analyzer_providers import (
            default_analyzer_providers,
        )

        analyzer_providers = default_analyzer_providers()
    if metric_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.metric_providers import (
            default_metric_providers,
        )

        metric_providers = default_metric_providers()
    if insight_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.insight_providers import (
            default_insight_providers,
        )

        insight_providers = default_insight_providers()
    if recommendation_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.recommendation_providers import (
            default_recommendation_providers,
        )

        recommendation_providers = default_recommendation_providers()
    return EnterpriseEvolutionIntelligenceEngine(
        intelligence_providers=intelligence_providers or (),
        analyzer_providers=analyzer_providers,
        metric_providers=metric_providers,
        insight_providers=insight_providers,
        recommendation_providers=recommendation_providers,
        audit_provider=audit_provider,
    )


__all__ = [
    "EnterpriseEvolutionIntelligenceEngine",
    "default_enterprise_evolution_intelligence_engine",
]
