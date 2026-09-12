# © Artur Czarnecki. All rights reserved.

"""Evolution impact analyzer provider plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionFutureScenario,
    EvolutionScenarioImpactAssessment,
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
)

_DEFAULT_IMPACT_ANALYZER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionImpactAnalyzerProvider:
    @property
    def provider_id(self) -> str:
        return "default_evolution_impact_analyzer"

    @property
    def provider_version(self) -> str:
        return _DEFAULT_IMPACT_ANALYZER_VERSION

    def assess(
        self,
        context: EvolutionStrategyContext,
        scenarios: tuple[EvolutionFutureScenario, ...],
    ) -> tuple[EvolutionScenarioImpactAssessment, ...]:
        assessments: list[EvolutionScenarioImpactAssessment] = []
        for scenario in scenarios:
            cost = "moderate"
            risk = "moderate"
            complexity = "moderate"
            value = "context_dependent"
            if "specialist" in scenario.scenario_id:
                cost = "potentially_lower_unit_cost"
                complexity = "higher_routing_complexity"
                value = "quality_specialization upside"
            elif "general" in scenario.scenario_id:
                cost = "higher_inference_cost"
                risk = "vendor_concentration"
                complexity = "lower_operational_complexity"
            elif "hybrid" in scenario.scenario_id:
                complexity = "highest_governance_overhead"
                value = "balanced_flexibility"
            refs = tuple(
                dict.fromkeys(
                    (
                        *scenario.data_source_refs,
                        EvolutionStrategyDataSourceRef(
                            source_id=scenario.scenario_id,
                            source_kind="future_scenario",
                            description=scenario.title,
                        ),
                    )
                )
            )
            assessments.append(
                EvolutionScenarioImpactAssessment(
                    assessment_id=f"impact:{scenario.scenario_id}",
                    scenario_id=scenario.scenario_id,
                    impact_analyzer_id=self.provider_id,
                    impact_analyzer_version=self.provider_version,
                    cost_assessment=cost,
                    risk_assessment=risk,
                    complexity_assessment=complexity,
                    value_assessment=value,
                    data_source_refs=refs,
                )
            )
        return tuple(assessments)


def default_impact_analyzer_providers() -> tuple[
    DefaultEvolutionImpactAnalyzerProvider,
]:
    return (DefaultEvolutionImpactAnalyzerProvider(),)


__all__ = [
    "DefaultEvolutionImpactAnalyzerProvider",
    "default_impact_analyzer_providers",
]
