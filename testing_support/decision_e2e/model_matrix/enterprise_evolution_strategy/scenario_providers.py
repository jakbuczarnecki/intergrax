# © Artur Czarnecki. All rights reserved.

"""Evolution scenario provider plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionFutureScenario,
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
    EvolutionStrategyDirectionFinding,
)

_DEFAULT_SCENARIO_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionScenarioProvider:
    @property
    def provider_id(self) -> str:
        return "default_evolution_scenario"

    @property
    def provider_version(self) -> str:
        return _DEFAULT_SCENARIO_PROVIDER_VERSION

    def build_scenarios(
        self,
        context: EvolutionStrategyContext,
        findings: tuple[EvolutionStrategyDirectionFinding, ...],
    ) -> tuple[EvolutionFutureScenario, ...]:
        finding_ids = tuple(item.finding_id for item in findings)
        base_refs = tuple(
            dict.fromkeys(
                ref for finding in findings for ref in finding.data_source_refs
            )
        )
        scope_ref = EvolutionStrategyDataSourceRef(
            source_id=f"scope:{context.scope_id}:{context.version}",
            source_kind="strategy_scope",
            description="Strategic analysis scope identifier.",
        )
        refs = tuple(dict.fromkeys((*base_refs, scope_ref)))
        return (
            EvolutionFutureScenario(
                scenario_id=f"scenario:specialist:{context.scope_id}",
                title="Greater specialist model utilization",
                summary=(
                    "Explore routing more workloads to specialist models "
                    "as capability quality improves."
                ),
                scenario_provider_id=self.provider_id,
                scenario_provider_version=self.provider_version,
                linked_finding_ids=finding_ids,
                data_source_refs=refs,
            ),
            EvolutionFutureScenario(
                scenario_id=f"scenario:general:{context.scope_id}",
                title="Continued general-purpose model emphasis",
                summary=(
                    "Maintain emphasis on larger general models where "
                    "integration simplicity dominates."
                ),
                scenario_provider_id=self.provider_id,
                scenario_provider_version=self.provider_version,
                linked_finding_ids=finding_ids,
                data_source_refs=refs,
            ),
            EvolutionFutureScenario(
                scenario_id=f"scenario:hybrid:{context.scope_id}",
                title="Hybrid routing portfolio",
                summary=(
                    "Balance specialist and general models via governed "
                    "routing policies — selection remains human-led."
                ),
                scenario_provider_id=self.provider_id,
                scenario_provider_version=self.provider_version,
                linked_finding_ids=finding_ids,
                data_source_refs=refs,
            ),
        )


def default_scenario_providers() -> tuple[DefaultEvolutionScenarioProvider,]:
    return (DefaultEvolutionScenarioProvider(),)


__all__ = [
    "DefaultEvolutionScenarioProvider",
    "default_scenario_providers",
]
