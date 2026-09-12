# © Artur Czarnecki. All rights reserved.

"""Evolution insight provider plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionAnalysisFinding,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
    EvolutionIntelligenceInsight,
    EvolutionMetricSnapshot,
)

_DEFAULT_INSIGHT_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionInsightProvider:
    @property
    def provider_id(self) -> str:
        return "default_evolution_insight"

    @property
    def provider_version(self) -> str:
        return _DEFAULT_INSIGHT_PROVIDER_VERSION

    def derive(
        self,
        context: EvolutionIntelligenceContext,
        findings: tuple[EvolutionAnalysisFinding, ...],
        metrics: tuple[EvolutionMetricSnapshot, ...],
    ) -> tuple[EvolutionIntelligenceInsight, ...]:
        if not findings:
            return ()
        effectiveness = next(
            (item for item in findings if item.category == "effectiveness"),
            findings[0],
        )
        cost = next((item for item in findings if item.category == "cost"), None)
        evidence_refs: list[EvolutionIntelligenceDataSourceRef] = []
        for item in findings:
            evidence_refs.extend(item.data_source_refs)
        for item in metrics:
            evidence_refs.extend(item.data_source_refs)
        deduped = tuple(dict.fromkeys(evidence_refs))
        observation = effectiveness.summary
        if cost is not None and "elevated" in cost.summary:
            observation = (
                f"{observation} Cost impact may be elevated; review recommended."
            )
        return (
            EvolutionIntelligenceInsight(
                insight_id=f"insight:{context.adaptation_id}",
                observation=observation,
                insight_provider_id=self.provider_id,
                insight_provider_version=self.provider_version,
                linked_finding_ids=tuple(item.finding_id for item in findings),
                linked_metric_ids=tuple(item.metric_id for item in metrics),
                evidence_source_refs=deduped,
            ),
        )


def default_insight_providers() -> tuple[DefaultEvolutionInsightProvider,]:
    return (DefaultEvolutionInsightProvider(),)


__all__ = [
    "DefaultEvolutionInsightProvider",
    "default_insight_providers",
]
