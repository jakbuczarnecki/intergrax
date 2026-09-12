# © Artur Czarnecki. All rights reserved.

"""Evolution intelligence audit metadata providers (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
    ENTERPRISE_EVOLUTION_INTELLIGENCE_VERSION,
    EvolutionIntelligenceAuditMetadata,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
)

_STANDARD_INTELLIGENCE_AUDIT_PROVIDER_ID = "standard_evolution_intelligence_audit"
_STANDARD_INTELLIGENCE_AUDIT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class StandardEvolutionIntelligenceAuditProvider:
    @property
    def provider_id(self) -> str:
        return _STANDARD_INTELLIGENCE_AUDIT_PROVIDER_ID

    @property
    def provider_version(self) -> str:
        return _STANDARD_INTELLIGENCE_AUDIT_PROVIDER_VERSION

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
    ) -> EvolutionIntelligenceAuditMetadata:
        return EvolutionIntelligenceAuditMetadata(
            intelligence_task_id=ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
            intelligence_layer_version=ENTERPRISE_EVOLUTION_INTELLIGENCE_VERSION,
            adaptation_id=context.adaptation_id,
            adaptation_version=context.version,
            history_reference=context.history_reference,
            intelligence_provider_ids=intelligence_provider_ids,
            intelligence_provider_versions=intelligence_provider_versions,
            analyzer_ids=analyzer_ids,
            analyzer_versions=analyzer_versions,
            metric_provider_ids=metric_provider_ids,
            metric_provider_versions=metric_provider_versions,
            insight_provider_ids=insight_provider_ids,
            insight_provider_versions=insight_provider_versions,
            recommendation_provider_ids=recommendation_provider_ids,
            recommendation_provider_versions=recommendation_provider_versions,
            data_source_refs=data_source_refs,
            analysis_scope_summary=analysis_scope_summary,
            analyzed_at=analyzed_at,
        )


def default_evolution_intelligence_audit_provider() -> (
    StandardEvolutionIntelligenceAuditProvider
):
    return StandardEvolutionIntelligenceAuditProvider()


__all__ = [
    "StandardEvolutionIntelligenceAuditProvider",
    "default_evolution_intelligence_audit_provider",
]
