# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence import (
    EvolutionGovernanceReference,
    EvolutionIntelligenceRunStatus,
    default_enterprise_evolution_intelligence_engine,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy import (
    EnterpriseEvolutionStrategyEngine,
    EvolutionCapabilityObservation,
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
    EvolutionStrategyDirectionFinding,
    EvolutionStrategyRecommendationKind,
    EvolutionStrategyRunStatus,
    default_enterprise_evolution_strategy_engine,
    default_enterprise_evolution_strategy_provider,
    default_evolution_strategy_audit_provider,
    default_impact_analyzer_providers,
    default_scenario_providers,
    default_strategy_recommendation_providers,
)
from testing_support.decision_e2e.model_matrix.self_improvement_governance.contracts import (
    SelfImprovementGovernanceStatus,
)
from tests.unit.testing_support.decision_e2e.model_matrix.test_enterprise_evolution_intelligence import (
    _context as intelligence_context,
    _stamp,
)


def _capability_observation() -> EvolutionCapabilityObservation:
    return EvolutionCapabilityObservation(
        observation_id="cap-obs-1",
        capability_name="specialist_routing",
        trend_direction="increasing",
        summary="Specialist model usage is rising.",
        data_source_refs=(
            EvolutionStrategyDataSourceRef(
                source_id="cap:usage:1",
                source_kind="capability_metric",
                description="Usage telemetry for specialist models.",
            ),
        ),
    )


def _strategy_context() -> EvolutionStrategyContext:
    intel_ctx = intelligence_context()
    intelligence = default_enterprise_evolution_intelligence_engine().analyze(
        intel_ctx,
        analyzed_at=_stamp(),
    )
    assert intelligence.status is EvolutionIntelligenceRunStatus.COMPLETE
    return EvolutionStrategyContext(
        scope_id="enterprise-scope-1",
        version="1",
        intelligence_result=intelligence,
        operation_records=intel_ctx.operation_records,
        execution_results=intel_ctx.execution_results,
        adaptation_history_refs=(intel_ctx.history_reference,),
        governance_reference=EvolutionGovernanceReference(
            governance_decision_reference="gov-decision-1",
            governance_status=SelfImprovementGovernanceStatus.APPROVED,
            approval_id="gov-appr-1",
        ),
        governance_decision=None,
        capability_observations=(_capability_observation(),),
        strategic_constraints=(),
    )


def test_default_strategy_provider_analyzes_evolution_history() -> None:
    provider = default_enterprise_evolution_strategy_provider()
    result = provider.analyze(_strategy_context(), analyzed_at=_stamp())
    assert result.status is EvolutionStrategyRunStatus.COMPLETE
    assert result.direction_findings
    assert any(
        item.analyzer_id == "capability_trend_analyzer"
        for item in result.direction_findings
    )


@dataclass(frozen=True, slots=True)
class CustomStrategyAnalyzerProvider:
    @property
    def provider_id(self) -> str:
        return "custom_strategy_analyzer"

    @property
    def provider_version(self) -> str:
        return "9"

    def analyze(
        self,
        context: EvolutionStrategyContext,
    ) -> tuple[EvolutionStrategyDirectionFinding, ...]:
        return (
            EvolutionStrategyDirectionFinding(
                finding_id="finding:custom-strategy",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                direction_theme="custom",
                summary=f"Custom strategic view for {context.scope_id}.",
                data_source_refs=(
                    EvolutionStrategyDataSourceRef(
                        source_id="custom:strategy:source",
                        source_kind="custom",
                        description="Custom strategy analyzer source.",
                    ),
                ),
            ),
        )


def test_analyzer_swap_without_engine_change() -> None:
    engine = EnterpriseEvolutionStrategyEngine(
        strategy_providers=(),
        strategy_analyzer_providers=(CustomStrategyAnalyzerProvider(),),
        scenario_providers=default_scenario_providers(),
        impact_analyzer_providers=default_impact_analyzer_providers(),
        recommendation_providers=default_strategy_recommendation_providers(),
        audit_provider=default_evolution_strategy_audit_provider(),
    )
    result = engine.analyze(_strategy_context(), analyzed_at=_stamp())
    assert any(
        item.analyzer_id == "custom_strategy_analyzer"
        for item in result.direction_findings
    )


def test_scenarios_include_sources() -> None:
    engine = default_enterprise_evolution_strategy_engine()
    result = engine.analyze(_strategy_context(), analyzed_at=_stamp())
    assert result.scenarios
    scenario = result.scenarios[0]
    assert scenario.data_source_refs
    assert scenario.linked_finding_ids


def test_recommendation_generation_without_execution() -> None:
    engine = default_enterprise_evolution_strategy_engine()
    result = engine.analyze(_strategy_context(), analyzed_at=_stamp())
    assert result.recommendations
    recommendation = result.recommendations[0]
    assert recommendation.kind in (
        EvolutionStrategyRecommendationKind.EXPLORE,
        EvolutionStrategyRecommendationKind.REVIEW,
        EvolutionStrategyRecommendationKind.COLLECT_MORE_DATA,
        EvolutionStrategyRecommendationKind.NO_ACTION,
    )
    assert recommendation.linked_scenario_ids
    assert "automatic" not in recommendation.summary.lower()


def test_strategy_audit_records_plugins_sources_timestamp() -> None:
    engine = default_enterprise_evolution_strategy_engine()
    result = engine.analyze(_strategy_context(), analyzed_at=_stamp())
    audit = result.audit
    assert audit.strategy_analyzer_ids
    assert "capability_trend_analyzer" in audit.strategy_analyzer_ids
    assert audit.scenario_provider_ids
    assert audit.recommendation_provider_ids
    assert audit.data_source_refs
    assert audit.analyzed_at == _stamp()
    assert audit.scope_id == "enterprise-scope-1"
