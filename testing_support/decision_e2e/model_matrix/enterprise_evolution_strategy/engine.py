# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution strategy orchestration via injected plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass
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
    EnterpriseEvolutionStrategyProvider,
    EvolutionImpactAnalyzerProvider,
    EvolutionScenarioProvider,
    EvolutionStrategyAnalyzerProvider,
    EvolutionStrategyAuditProvider,
    EvolutionStrategyRecommendationProvider,
)


def _has_strategy_input(context: EvolutionStrategyContext) -> bool:
    return bool(
        context.intelligence_result is not None
        or context.operation_records
        or context.execution_results
        or context.adaptation_history_refs
        or context.capability_observations
    )


def _provider_ids(
    providers: tuple[
        EnterpriseEvolutionStrategyProvider
        | EvolutionStrategyAnalyzerProvider
        | EvolutionScenarioProvider
        | EvolutionImpactAnalyzerProvider
        | EvolutionStrategyRecommendationProvider,
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_id for item in providers)


def _provider_versions(
    providers: tuple[
        EnterpriseEvolutionStrategyProvider
        | EvolutionStrategyAnalyzerProvider
        | EvolutionScenarioProvider
        | EvolutionImpactAnalyzerProvider
        | EvolutionStrategyRecommendationProvider,
        ...,
    ],
) -> tuple[str, ...]:
    return tuple(item.provider_version for item in providers)


@dataclass(frozen=True, slots=True)
class EnterpriseEvolutionStrategyEngine:
    strategy_providers: tuple[EnterpriseEvolutionStrategyProvider, ...]
    strategy_analyzer_providers: tuple[EvolutionStrategyAnalyzerProvider, ...]
    scenario_providers: tuple[EvolutionScenarioProvider, ...]
    impact_analyzer_providers: tuple[EvolutionImpactAnalyzerProvider, ...]
    recommendation_providers: tuple[EvolutionStrategyRecommendationProvider, ...]
    audit_provider: EvolutionStrategyAuditProvider

    def analyze(
        self,
        context: EvolutionStrategyContext,
        *,
        analyzed_at: datetime | None = None,
    ) -> EvolutionStrategyResult:
        stamp = analyzed_at or datetime.now(tz=UTC)
        empty_audit = self.audit_provider.build_audit(
            context,
            strategy_provider_ids=_provider_ids(self.strategy_providers),
            strategy_provider_versions=_provider_versions(self.strategy_providers),
            strategy_analyzer_ids=_provider_ids(self.strategy_analyzer_providers),
            strategy_analyzer_versions=_provider_versions(
                self.strategy_analyzer_providers
            ),
            scenario_provider_ids=_provider_ids(self.scenario_providers),
            scenario_provider_versions=_provider_versions(self.scenario_providers),
            impact_analyzer_ids=_provider_ids(self.impact_analyzer_providers),
            impact_analyzer_versions=_provider_versions(self.impact_analyzer_providers),
            recommendation_provider_ids=_provider_ids(self.recommendation_providers),
            recommendation_provider_versions=_provider_versions(
                self.recommendation_providers
            ),
            scenario_ids=(),
            recommendation_ids=(),
            data_source_refs=(),
            analyzed_at=stamp,
            analysis_scope_summary="No strategic evolution context supplied.",
        )
        if not _has_strategy_input(context):
            return EvolutionStrategyResult(
                strategy_task_id=ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
                status=EvolutionStrategyRunStatus.INSUFFICIENT_INPUT,
                audit=empty_audit,
                direction_findings=(),
                scenarios=(),
                impact_assessments=(),
                recommendations=(),
            )

        findings, scenarios, impacts, recommendations, data_refs = (
            run_evolution_strategy_pipeline(
                context,
                strategy_analyzer_providers=self.strategy_analyzer_providers,
                scenario_providers=self.scenario_providers,
                impact_analyzer_providers=self.impact_analyzer_providers,
                recommendation_providers=self.recommendation_providers,
            )
        )

        supplemental_findings = findings
        supplemental_scenarios = scenarios
        supplemental_impacts = impacts
        supplemental_recommendations = recommendations
        supplemental_refs = data_refs

        for provider in self.strategy_providers:
            partial = provider.analyze(context, analyzed_at=stamp)
            supplemental_findings = (
                *supplemental_findings,
                *partial.direction_findings,
            )
            supplemental_scenarios = (*supplemental_scenarios, *partial.scenarios)
            supplemental_impacts = (
                *supplemental_impacts,
                *partial.impact_assessments,
            )
            supplemental_recommendations = (
                *supplemental_recommendations,
                *partial.recommendations,
            )
            supplemental_refs = tuple(
                dict.fromkeys((*supplemental_refs, *partial.audit.data_source_refs))
            )

        scope_summary = (
            f"Strategic analysis for scope {context.scope_id} v{context.version} "
            f"with {len(context.operation_records)} operations, "
            f"{len(context.execution_results)} executions, "
            f"{len(context.capability_observations)} capability observations."
        )
        audit = self.audit_provider.build_audit(
            context,
            strategy_provider_ids=_provider_ids(self.strategy_providers),
            strategy_provider_versions=_provider_versions(self.strategy_providers),
            strategy_analyzer_ids=_provider_ids(self.strategy_analyzer_providers),
            strategy_analyzer_versions=_provider_versions(
                self.strategy_analyzer_providers
            ),
            scenario_provider_ids=_provider_ids(self.scenario_providers),
            scenario_provider_versions=_provider_versions(self.scenario_providers),
            impact_analyzer_ids=_provider_ids(self.impact_analyzer_providers),
            impact_analyzer_versions=_provider_versions(self.impact_analyzer_providers),
            recommendation_provider_ids=_provider_ids(self.recommendation_providers),
            recommendation_provider_versions=_provider_versions(
                self.recommendation_providers
            ),
            scenario_ids=tuple(item.scenario_id for item in supplemental_scenarios),
            recommendation_ids=tuple(
                item.recommendation_id for item in supplemental_recommendations
            ),
            data_source_refs=supplemental_refs,
            analyzed_at=stamp,
            analysis_scope_summary=scope_summary,
        )
        status = (
            EvolutionStrategyRunStatus.COMPLETE
            if supplemental_findings
            or supplemental_scenarios
            or supplemental_recommendations
            else EvolutionStrategyRunStatus.INSUFFICIENT_INPUT
        )
        return EvolutionStrategyResult(
            strategy_task_id=ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
            status=status,
            audit=audit,
            direction_findings=supplemental_findings,
            scenarios=supplemental_scenarios,
            impact_assessments=supplemental_impacts,
            recommendations=supplemental_recommendations,
        )


def default_enterprise_evolution_strategy_engine(
    *,
    strategy_providers: tuple[EnterpriseEvolutionStrategyProvider, ...] | None = None,
    strategy_analyzer_providers: tuple[EvolutionStrategyAnalyzerProvider, ...]
    | None = None,
    scenario_providers: tuple[EvolutionScenarioProvider, ...] | None = None,
    impact_analyzer_providers: tuple[EvolutionImpactAnalyzerProvider, ...]
    | None = None,
    recommendation_providers: tuple[EvolutionStrategyRecommendationProvider, ...]
    | None = None,
    audit_provider: EvolutionStrategyAuditProvider | None = None,
) -> EnterpriseEvolutionStrategyEngine:
    if audit_provider is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.audit_providers import (
            default_evolution_strategy_audit_provider,
        )

        audit_provider = default_evolution_strategy_audit_provider()
    if strategy_analyzer_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.analyzer_providers import (
            default_strategy_analyzer_providers,
        )

        strategy_analyzer_providers = default_strategy_analyzer_providers()
    if scenario_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.scenario_providers import (
            default_scenario_providers,
        )

        scenario_providers = default_scenario_providers()
    if impact_analyzer_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.impact_providers import (
            default_impact_analyzer_providers,
        )

        impact_analyzer_providers = default_impact_analyzer_providers()
    if recommendation_providers is None:
        from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.recommendation_providers import (
            default_strategy_recommendation_providers,
        )

        recommendation_providers = default_strategy_recommendation_providers()
    return EnterpriseEvolutionStrategyEngine(
        strategy_providers=strategy_providers or (),
        strategy_analyzer_providers=strategy_analyzer_providers,
        scenario_providers=scenario_providers,
        impact_analyzer_providers=impact_analyzer_providers,
        recommendation_providers=recommendation_providers,
        audit_provider=audit_provider,
    )


__all__ = [
    "EnterpriseEvolutionStrategyEngine",
    "default_enterprise_evolution_strategy_engine",
]
