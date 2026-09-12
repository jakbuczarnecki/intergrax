# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution strategy provider plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
    EvolutionStrategyContext,
    EvolutionStrategyResult,
    EvolutionStrategyRunStatus,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.pipeline import (
    run_evolution_strategy_pipeline,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.protocol import (
    EvolutionImpactAnalyzerProvider,
    EvolutionScenarioProvider,
    EvolutionStrategyAnalyzerProvider,
    EvolutionStrategyAuditProvider,
    EvolutionStrategyRecommendationProvider,
)

_DEFAULT_STRATEGY_PROVIDER_ID = "default_enterprise_evolution_strategy"
_DEFAULT_STRATEGY_PROVIDER_VERSION = "1"


@dataclass
class DefaultEnterpriseEvolutionStrategyProvider:
    """Default strategy plugin — composes analyzer/scenario/impact/recommendation plugins."""

    strategy_analyzer_providers: tuple[EvolutionStrategyAnalyzerProvider, ...] = field(
        default_factory=tuple
    )
    scenario_providers: tuple[EvolutionScenarioProvider, ...] = field(
        default_factory=tuple
    )
    impact_analyzer_providers: tuple[EvolutionImpactAnalyzerProvider, ...] = field(
        default_factory=tuple
    )
    recommendation_providers: tuple[EvolutionStrategyRecommendationProvider, ...] = (
        field(default_factory=tuple)
    )
    audit_provider: EvolutionStrategyAuditProvider | None = None

    @property
    def provider_id(self) -> str:
        return _DEFAULT_STRATEGY_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _DEFAULT_STRATEGY_PROVIDER_VERSION

    def analyze(
        self,
        context: EvolutionStrategyContext,
        *,
        analyzed_at: datetime | None = None,
    ) -> EvolutionStrategyResult:
        stamp = analyzed_at or datetime.now(tz=UTC)
        analyzers = self._resolved_analyzers()
        scenarios_p = self._resolved_scenarios()
        impacts = self._resolved_impacts()
        recommendations = self._resolved_recommendations()
        audit_provider = self._resolved_audit()

        if not _has_strategy_input(context):
            audit = audit_provider.build_audit(
                context,
                strategy_provider_ids=(self.provider_id,),
                strategy_provider_versions=(self.provider_version,),
                strategy_analyzer_ids=tuple(item.provider_id for item in analyzers),
                strategy_analyzer_versions=tuple(
                    item.provider_version for item in analyzers
                ),
                scenario_provider_ids=tuple(item.provider_id for item in scenarios_p),
                scenario_provider_versions=tuple(
                    item.provider_version for item in scenarios_p
                ),
                impact_analyzer_ids=tuple(item.provider_id for item in impacts),
                impact_analyzer_versions=tuple(
                    item.provider_version for item in impacts
                ),
                recommendation_provider_ids=tuple(
                    item.provider_id for item in recommendations
                ),
                recommendation_provider_versions=tuple(
                    item.provider_version for item in recommendations
                ),
                scenario_ids=(),
                recommendation_ids=(),
                data_source_refs=(),
                analyzed_at=stamp,
                analysis_scope_summary="Insufficient strategic context for analysis.",
            )
            return EvolutionStrategyResult(
                strategy_task_id=ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
                status=EvolutionStrategyRunStatus.INSUFFICIENT_INPUT,
                audit=audit,
                direction_findings=(),
                scenarios=(),
                impact_assessments=(),
                recommendations=(),
            )

        findings, scenarios, impact_rows, recommendation_rows, data_refs = (
            run_evolution_strategy_pipeline(
                context,
                strategy_analyzer_providers=analyzers,
                scenario_providers=scenarios_p,
                impact_analyzer_providers=impacts,
                recommendation_providers=recommendations,
            )
        )
        scope_summary = (
            f"Default strategy analysis for scope {context.scope_id} "
            f"v{context.version}."
        )
        audit = audit_provider.build_audit(
            context,
            strategy_provider_ids=(self.provider_id,),
            strategy_provider_versions=(self.provider_version,),
            strategy_analyzer_ids=tuple(item.provider_id for item in analyzers),
            strategy_analyzer_versions=tuple(
                item.provider_version for item in analyzers
            ),
            scenario_provider_ids=tuple(item.provider_id for item in scenarios_p),
            scenario_provider_versions=tuple(
                item.provider_version for item in scenarios_p
            ),
            impact_analyzer_ids=tuple(item.provider_id for item in impacts),
            impact_analyzer_versions=tuple(item.provider_version for item in impacts),
            recommendation_provider_ids=tuple(
                item.provider_id for item in recommendations
            ),
            recommendation_provider_versions=tuple(
                item.provider_version for item in recommendations
            ),
            scenario_ids=tuple(item.scenario_id for item in scenarios),
            recommendation_ids=tuple(
                item.recommendation_id for item in recommendation_rows
            ),
            data_source_refs=data_refs,
            analyzed_at=stamp,
            analysis_scope_summary=scope_summary,
        )
        return EvolutionStrategyResult(
            strategy_task_id=ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
            status=EvolutionStrategyRunStatus.COMPLETE,
            audit=audit,
            direction_findings=findings,
            scenarios=scenarios,
            impact_assessments=impact_rows,
            recommendations=recommendation_rows,
        )

    def _resolved_analyzers(self) -> tuple[EvolutionStrategyAnalyzerProvider, ...]:
        if self.strategy_analyzer_providers:
            return self.strategy_analyzer_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.analyzer_providers import (
            default_strategy_analyzer_providers,
        )

        return default_strategy_analyzer_providers()

    def _resolved_scenarios(self) -> tuple[EvolutionScenarioProvider, ...]:
        if self.scenario_providers:
            return self.scenario_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.scenario_providers import (
            default_scenario_providers,
        )

        return default_scenario_providers()

    def _resolved_impacts(self) -> tuple[EvolutionImpactAnalyzerProvider, ...]:
        if self.impact_analyzer_providers:
            return self.impact_analyzer_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.impact_providers import (
            default_impact_analyzer_providers,
        )

        return default_impact_analyzer_providers()

    def _resolved_recommendations(
        self,
    ) -> tuple[EvolutionStrategyRecommendationProvider, ...]:
        if self.recommendation_providers:
            return self.recommendation_providers
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.recommendation_providers import (
            default_strategy_recommendation_providers,
        )

        return default_strategy_recommendation_providers()

    def _resolved_audit(self) -> EvolutionStrategyAuditProvider:
        if self.audit_provider is not None:
            return self.audit_provider
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.audit_providers import (
            default_evolution_strategy_audit_provider,
        )

        return default_evolution_strategy_audit_provider()


def _has_strategy_input(context: EvolutionStrategyContext) -> bool:
    return bool(
        context.intelligence_result is not None
        or context.operation_records
        or context.execution_results
        or context.adaptation_history_refs
        or context.capability_observations
    )


def default_enterprise_evolution_strategy_provider() -> (
    DefaultEnterpriseEvolutionStrategyProvider
):
    return DefaultEnterpriseEvolutionStrategyProvider()


__all__ = [
    "DefaultEnterpriseEvolutionStrategyProvider",
    "default_enterprise_evolution_strategy_provider",
]
