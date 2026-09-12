# © Artur Czarnecki. All rights reserved.

"""Evolution strategy analyzer provider plugins (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
    EvolutionStrategyDirectionFinding,
)

_ANALYZER_VERSION = "1"


def _intelligence_source(
    context: EvolutionStrategyContext,
) -> EvolutionStrategyDataSourceRef | None:
    if context.intelligence_result is None:
        return None
    return EvolutionStrategyDataSourceRef(
        source_id=f"intelligence:{context.scope_id}:{context.version}",
        source_kind="evolution_intelligence_result",
        description="Prior evolution intelligence analysis used for strategic context.",
    )


@dataclass(frozen=True, slots=True)
class CapabilityTrendAnalyzer:
    @property
    def provider_id(self) -> str:
        return "capability_trend_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionStrategyContext,
    ) -> tuple[EvolutionStrategyDirectionFinding, ...]:
        refs: list[EvolutionStrategyDataSourceRef] = []
        intel = _intelligence_source(context)
        if intel is not None:
            refs.append(intel)
        for obs in context.capability_observations:
            refs.extend(obs.data_source_refs)
        if not context.capability_observations and intel is None:
            return ()
        rising = sum(
            1
            for item in context.capability_observations
            if item.trend_direction == "increasing"
        )
        summary = (
            f"Capability trends: {rising} increasing of "
            f"{len(context.capability_observations)} observations."
        )
        if rising > 0:
            summary += " Specialist capability usage may warrant strategic exploration."
        return (
            EvolutionStrategyDirectionFinding(
                finding_id=f"finding:capability:{context.scope_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                direction_theme="capability_evolution",
                summary=summary,
                data_source_refs=tuple(dict.fromkeys(refs)),
            ),
        )


@dataclass(frozen=True, slots=True)
class CostEvolutionAnalyzer:
    @property
    def provider_id(self) -> str:
        return "cost_evolution_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionStrategyContext,
    ) -> tuple[EvolutionStrategyDirectionFinding, ...]:
        refs: list[EvolutionStrategyDataSourceRef] = []
        intel = _intelligence_source(context)
        if intel is not None:
            refs.append(intel)
        for record in context.operation_records:
            refs.append(
                EvolutionStrategyDataSourceRef(
                    source_id=f"operation:{record.record_id}",
                    source_kind="evolution_operation_record",
                    description=record.summary,
                )
            )
        if not refs:
            return ()
        return (
            EvolutionStrategyDirectionFinding(
                finding_id=f"finding:cost:{context.scope_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                direction_theme="cost_evolution",
                summary=(
                    "Operational history suggests monitoring cost trajectories "
                    "when scaling model usage."
                ),
                data_source_refs=tuple(dict.fromkeys(refs)),
            ),
        )


@dataclass(frozen=True, slots=True)
class RiskEvolutionAnalyzer:
    @property
    def provider_id(self) -> str:
        return "risk_evolution_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionStrategyContext,
    ) -> tuple[EvolutionStrategyDirectionFinding, ...]:
        refs: list[EvolutionStrategyDataSourceRef] = []
        if context.governance_reference is not None:
            refs.append(
                EvolutionStrategyDataSourceRef(
                    source_id=context.governance_reference.governance_decision_reference,
                    source_kind="governance_reference",
                    description="Governance trace for strategic risk context.",
                )
            )
        if not context.execution_results and not refs:
            return ()
        return (
            EvolutionStrategyDirectionFinding(
                finding_id=f"finding:risk:{context.scope_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                direction_theme="risk_evolution",
                summary=(
                    "Governed execution history informs long-term risk posture; "
                    "no automatic risk acceptance."
                ),
                data_source_refs=tuple(dict.fromkeys(refs)),
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchitectureTrendAnalyzer:
    @property
    def provider_id(self) -> str:
        return "architecture_trend_analyzer"

    @property
    def provider_version(self) -> str:
        return _ANALYZER_VERSION

    def analyze(
        self,
        context: EvolutionStrategyContext,
    ) -> tuple[EvolutionStrategyDirectionFinding, ...]:
        refs: list[EvolutionStrategyDataSourceRef] = []
        for hist in context.adaptation_history_refs:
            refs.append(
                EvolutionStrategyDataSourceRef(
                    source_id=f"history:{hist.adaptation_id}:{hist.version}",
                    source_kind="adaptation_history",
                    description="Historical adaptation reference for architecture trends.",
                )
            )
        intel = _intelligence_source(context)
        if intel is not None:
            refs.append(intel)
        if not refs:
            return ()
        return (
            EvolutionStrategyDirectionFinding(
                finding_id=f"finding:architecture:{context.scope_id}",
                analyzer_id=self.provider_id,
                analyzer_version=self.provider_version,
                direction_theme="architecture_evolution",
                summary=(
                    "Adaptation history indicates evolving architecture patterns; "
                    "humans retain architecture authority."
                ),
                data_source_refs=tuple(dict.fromkeys(refs)),
            ),
        )


def default_strategy_analyzer_providers() -> tuple[
    CapabilityTrendAnalyzer,
    CostEvolutionAnalyzer,
    RiskEvolutionAnalyzer,
    ArchitectureTrendAnalyzer,
]:
    return (
        CapabilityTrendAnalyzer(),
        CostEvolutionAnalyzer(),
        RiskEvolutionAnalyzer(),
        ArchitectureTrendAnalyzer(),
    )


__all__ = [
    "ArchitectureTrendAnalyzer",
    "CapabilityTrendAnalyzer",
    "CostEvolutionAnalyzer",
    "RiskEvolutionAnalyzer",
    "default_strategy_analyzer_providers",
]
