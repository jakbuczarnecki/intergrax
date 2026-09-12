# © Artur Czarnecki. All rights reserved.

"""Pluggable enterprise evolution intelligence contracts (DS-E2E-15J-L15)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionAnalysisFinding,
    EvolutionIntelligenceAuditMetadata,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
    EvolutionIntelligenceInsight,
    EvolutionIntelligenceRecommendation,
    EvolutionIntelligenceResult,
    EvolutionMetricSnapshot,
)


class EnterpriseEvolutionIntelligenceProvider(Protocol):
    """Pluggable evolution intelligence — analyzes history, never mutates runtime."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
        *,
        analyzed_at: datetime | None = None,
    ) -> EvolutionIntelligenceResult: ...


class EvolutionAnalyzerProvider(Protocol):
    """Pluggable analyzer over evolution context — returns facts, not actions."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def analyze(
        self,
        context: EvolutionIntelligenceContext,
    ) -> tuple[EvolutionAnalysisFinding, ...]: ...


class EvolutionMetricProvider(Protocol):
    """Pluggable metric extraction — descriptive measurements only."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def collect(
        self,
        context: EvolutionIntelligenceContext,
        findings: tuple[EvolutionAnalysisFinding, ...],
    ) -> tuple[EvolutionMetricSnapshot, ...]: ...


class EvolutionInsightProvider(Protocol):
    """Transforms analysis facts into observations with traceable evidence."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def derive(
        self,
        context: EvolutionIntelligenceContext,
        findings: tuple[EvolutionAnalysisFinding, ...],
        metrics: tuple[EvolutionMetricSnapshot, ...],
    ) -> tuple[EvolutionIntelligenceInsight, ...]: ...


class EvolutionRecommendationProvider(Protocol):
    """Suggests maintenance follow-ups — recommendations never execute changes."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def recommend(
        self,
        context: EvolutionIntelligenceContext,
        insights: tuple[EvolutionIntelligenceInsight, ...],
    ) -> tuple[EvolutionIntelligenceRecommendation, ...]: ...


class EvolutionIntelligenceAuditProvider(Protocol):
    """Records auditable trace for every intelligence analysis run."""

    @property
    def provider_id(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def build_audit(
        self,
        context: EvolutionIntelligenceContext,
        *,
        intelligence_provider_ids: tuple[str, ...],
        intelligence_provider_versions: tuple[str, ...],
        analyzer_ids: tuple[str, ...],
        analyzer_versions: tuple[str, ...],
        metric_provider_ids: tuple[str, ...],
        metric_provider_versions: tuple[str, ...],
        insight_provider_ids: tuple[str, ...],
        insight_provider_versions: tuple[str, ...],
        recommendation_provider_ids: tuple[str, ...],
        recommendation_provider_versions: tuple[str, ...],
        data_source_refs: tuple[EvolutionIntelligenceDataSourceRef, ...],
        analyzed_at: datetime,
        analysis_scope_summary: str,
    ) -> EvolutionIntelligenceAuditMetadata: ...


__all__ = [
    "EnterpriseEvolutionIntelligenceProvider",
    "EvolutionAnalyzerProvider",
    "EvolutionInsightProvider",
    "EvolutionIntelligenceAuditProvider",
    "EvolutionMetricProvider",
    "EvolutionRecommendationProvider",
]
