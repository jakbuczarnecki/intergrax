# © Artur Czarnecki. All rights reserved.

"""Pluggable enterprise evolution strategy contracts (DS-E2E-15J-L16)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.enterprise_evolution_strategy.contracts import (
    EvolutionFutureScenario,
    EvolutionScenarioImpactAssessment,
    EvolutionStrategicRecommendation,
    EvolutionStrategyAuditMetadata,
    EvolutionStrategyContext,
    EvolutionStrategyDataSourceRef,
    EvolutionStrategyDirectionFinding,
    EvolutionStrategyResult,
)


class EnterpriseEvolutionStrategyProvider(Protocol):
    """Pluggable evolution strategy — analyzes directions, never decides or mutates."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze(
        self,
        context: EvolutionStrategyContext,
        *,
        analyzed_at: datetime | None = None,
    ) -> EvolutionStrategyResult: ...


class EvolutionStrategyAnalyzerProvider(Protocol):
    """Pluggable long-term direction analysis — facts only."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze(
        self,
        context: EvolutionStrategyContext,
    ) -> tuple[EvolutionStrategyDirectionFinding, ...]: ...


class EvolutionScenarioProvider(Protocol):
    """Produces candidate future scenarios — does not select a winner."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def build_scenarios(
        self,
        context: EvolutionStrategyContext,
        findings: tuple[EvolutionStrategyDirectionFinding, ...],
    ) -> tuple[EvolutionFutureScenario, ...]: ...


class EvolutionImpactAnalyzerProvider(Protocol):
    """Assesses scenario impacts across dimensions — no aggregate strategy score."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def assess(
        self,
        context: EvolutionStrategyContext,
        scenarios: tuple[EvolutionFutureScenario, ...],
    ) -> tuple[EvolutionScenarioImpactAssessment, ...]: ...


class EvolutionStrategyRecommendationProvider(Protocol):
    """Strategic suggestions for human governance — never executes changes."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def recommend(
        self,
        context: EvolutionStrategyContext,
        scenarios: tuple[EvolutionFutureScenario, ...],
        impact_assessments: tuple[EvolutionScenarioImpactAssessment, ...],
    ) -> tuple[EvolutionStrategicRecommendation, ...]: ...


class EvolutionStrategyAuditProvider(Protocol):
    """Records auditable trace for every strategy analysis run."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

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
    ) -> EvolutionStrategyAuditMetadata: ...


__all__ = [
    "EnterpriseEvolutionStrategyProvider",
    "EvolutionImpactAnalyzerProvider",
    "EvolutionScenarioProvider",
    "EvolutionStrategyAnalyzerProvider",
    "EvolutionStrategyAuditProvider",
    "EvolutionStrategyRecommendationProvider",
]
