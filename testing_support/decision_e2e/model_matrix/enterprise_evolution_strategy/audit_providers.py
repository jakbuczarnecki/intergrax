# © Artur Czarnecki. All rights reserved.

"""Evolution strategy audit metadata providers (DS-E2E-15J-L16)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
    ENTERPRISE_EVOLUTION_STRATEGY_VERSION,
    EvolutionStrategyAuditMetadata,
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
)

_STANDARD_STRATEGY_AUDIT_PROVIDER_ID = "standard_evolution_strategy_audit"
_STANDARD_STRATEGY_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardEvolutionStrategyAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_STRATEGY_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_STRATEGY_AUDIT_PROVIDER_VERSION

    def build_audit(
        self,
        context: EvolutionStrategyContext,
        *,
        strategy_provider_ids: tuple[str, ...],
        strategy_provider_versions: tuple[str, ...],
        strategy_analyzer_ids: tuple[str, ...],
        strategy_analyzer_versions: tuple[str, ...],
        scenario_provider_ids: tuple[str, ...],
        scenario_provider_versions: tuple[str, ...],
        impact_analyzer_ids: tuple[str, ...],
        impact_analyzer_versions: tuple[str, ...],
        recommendation_provider_ids: tuple[str, ...],
        recommendation_provider_versions: tuple[str, ...],
        scenario_ids: tuple[str, ...],
        recommendation_ids: tuple[str, ...],
        data_source_refs: tuple[EvolutionStrategyDataSourceRef, ...],
        analyzed_at: datetime,
        analysis_scope_summary: str,
    ) -> EvolutionStrategyAuditMetadata:
        return EvolutionStrategyAuditMetadata(
            strategy_task_id=ENTERPRISE_EVOLUTION_STRATEGY_TASK_ID,
            strategy_layer_version=ENTERPRISE_EVOLUTION_STRATEGY_VERSION,
            scope_id=context.scope_id,
            scope_version=context.version,
            strategy_provider_ids=strategy_provider_ids,
            strategy_provider_versions=strategy_provider_versions,
            strategy_analyzer_ids=strategy_analyzer_ids,
            strategy_analyzer_versions=strategy_analyzer_versions,
            scenario_provider_ids=scenario_provider_ids,
            scenario_provider_versions=scenario_provider_versions,
            impact_analyzer_ids=impact_analyzer_ids,
            impact_analyzer_versions=impact_analyzer_versions,
            recommendation_provider_ids=recommendation_provider_ids,
            recommendation_provider_versions=recommendation_provider_versions,
            scenario_ids=scenario_ids,
            recommendation_ids=recommendation_ids,
            data_source_refs=data_source_refs,
            analysis_scope_summary=analysis_scope_summary,
            analyzed_at=analyzed_at,
        )


def default_evolution_strategy_audit_provider() -> (
    StandardEvolutionStrategyAuditProvider
):
    return StandardEvolutionStrategyAuditProvider()


__all__ = [
    "StandardEvolutionStrategyAuditProvider",
    "default_evolution_strategy_audit_provider",
]
