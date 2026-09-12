# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution intelligence provider plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    EvolutionAnalyzerProvider,
    EvolutionInsightProvider,
    EvolutionIntelligenceAuditProvider,
    EvolutionMetricProvider,
    EvolutionRecommendationProvider,
)

_DEFAULT_INTELLIGENCE_PROVIDER_ID = "default_enterprise_evolution_intelligence"
_DEFAULT_INTELLIGENCE_PROVIDER_VERSION = "1"


@dataclass
class DefaultEnterpriseEvolutionIntelligenceProvider:
    """Default intelligence plugin — composes analyzer/metric/insight/recommendation plugins."""

    analyzer_providers: tuple[EvolutionAnalyzerProvider, ...] = field(
        default_factory=tuple
    )
    metric_providers: tuple[EvolutionMetricProvider, ...] = field(default_factory=tuple)
    insight_providers: tuple[EvolutionInsightProvider, ...] = field(
        default_factory=tuple
    )
    recommendation_providers: tuple[EvolutionRecommendationProvider, ...] = field(
        default_factory=tuple
    )
    audit_provider: EvolutionIntelligenceAuditProvider | None = None

    @property
    def provider_id(self) -> str:
        return _DEFAULT_INTELLIGENCE_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_INTELLIGENCE_PROVIDER_VERSION

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
        *,
        analyzed_at: datetime | None = None,
    ) -> EvolutionIntelligenceResult:
        stamp = analyzed_at or datetime.now(tz=UTC)
        analyzers = self._resolved_analyzers()
        metrics = self._resolved_metrics()
        insights = self._resolved_insights()
        recommendations = self._resolved_recommendations()
        audit_provider = self._resolved_audit()

        if not (
            context.operation_records
            or context.execution_results
            or context.health_observations
        ):
            audit = audit_provider.build_audit(
                context,
                intelligence_provider_ids=(self.provider_id,),
                intelligence_provider_versions=(self.provider_version,),
                analyzer_ids=tuple(item.provider_id for item in analyzers),
                analyzer_versions=tuple(item.provider_version for item in analyzers),
                metric_provider_ids=tuple(item.provider_id for item in metrics),
                metric_provider_versions=tuple(
                    item.provider_version for item in metrics
                ),
                insight_provider_ids=tuple(item.provider_id for item in insights),
                insight_provider_versions=tuple(
                    item.provider_version for item in insights
                ),
                recommendation_provider_ids=tuple(
                    item.provider_id for item in recommendations
                ),
                recommendation_provider_versions=tuple(
                    item.provider_version for item in recommendations
                ),
                data_source_refs=(),
                analyzed_at=stamp,
                analysis_scope_summary="Insufficient evolution history for analysis.",
            )
            return EvolutionIntelligenceResult(
                intelligence_task_id=ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
                status=EvolutionIntelligenceRunStatus.INSUFFICIENT_INPUT,
                audit=audit,
                analysis_findings=(),
                metric_snapshots=(),
                insights=(),
                recommendations=(),
            )

        findings, metric_snapshots, insight_rows, recommendation_rows, data_refs = (
            run_evolution_intelligence_pipeline(
                context,
                analyzer_providers=analyzers,
                metric_providers=metrics,
                insight_providers=insights,
                recommendation_providers=recommendations,
            )
        )
        scope_summary = (
            f"Default intelligence analysis for {context.adaptation_id} "
            f"v{context.version}."
        )
        audit = audit_provider.build_audit(
            context,
            intelligence_provider_ids=(self.provider_id,),
            intelligence_provider_versions=(self.provider_version,),
            analyzer_ids=tuple(item.provider_id for item in analyzers),
            analyzer_versions=tuple(item.provider_version for item in analyzers),
            metric_provider_ids=tuple(item.provider_id for item in metrics),
            metric_provider_versions=tuple(item.provider_version for item in metrics),
            insight_provider_ids=tuple(item.provider_id for item in insights),
            insight_provider_versions=tuple(item.provider_version for item in insights),
            recommendation_provider_ids=tuple(
                item.provider_id for item in recommendations
            ),
            recommendation_provider_versions=tuple(
                item.provider_version for item in recommendations
            ),
            data_source_refs=data_refs,
            analyzed_at=stamp,
            analysis_scope_summary=scope_summary,
        )
        return EvolutionIntelligenceResult(
            intelligence_task_id=ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
            status=EvolutionIntelligenceRunStatus.COMPLETE,
            audit=audit,
            analysis_findings=findings,
            metric_snapshots=metric_snapshots,
            insights=insight_rows,
            recommendations=recommendation_rows,
        )

    def _resolved_analyzers(self) -> tuple[EvolutionAnalyzerProvider, ...]:
        if self.analyzer_providers:
            return self.analyzer_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.analyzer_providers import (
            default_analyzer_providers,
        )

        return default_analyzer_providers()

    def _resolved_metrics(self) -> tuple[EvolutionMetricProvider, ...]:
        if self.metric_providers:
            return self.metric_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.metric_providers import (
            default_metric_providers,
        )

        return default_metric_providers()

    def _resolved_insights(self) -> tuple[EvolutionInsightProvider, ...]:
        if self.insight_providers:
            return self.insight_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.insight_providers import (
            default_insight_providers,
        )

        return default_insight_providers()

    def _resolved_recommendations(self) -> tuple[EvolutionRecommendationProvider, ...]:
        if self.recommendation_providers:
            return self.recommendation_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.recommendation_providers import (
            default_recommendation_providers,
        )

        return default_recommendation_providers()

    def _resolved_audit(self) -> EvolutionIntelligenceAuditProvider:
        if self.audit_provider is not None:
            return self.audit_provider
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.audit_providers import (
            default_evolution_intelligence_audit_provider,
        )

        return default_evolution_intelligence_audit_provider()


def default_enterprise_evolution_intelligence_provider() -> (
    DefaultEnterpriseEvolutionIntelligenceProvider
):
    return DefaultEnterpriseEvolutionIntelligenceProvider()


__all__ = [
    "DefaultEnterpriseEvolutionIntelligenceProvider",
    "default_enterprise_evolution_intelligence_provider",
]
