# © Artur Czarnecki. All rights reserved.

"""Evolution metric provider plugins (DS-E2E-15J-L15)."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    EvolutionAnalysisFinding,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
    EvolutionMetricSnapshot,
)

_DEFAULT_METRIC_PROVIDER_VERSION = "1"


@dataclass(frozen=True, slots=True)
class DefaultEvolutionMetricProvider:
    @property
    def provider_id(self) -> str:
        return "default_evolution_metric"

    @property
    def provider_version(self) -> str:
        return _DEFAULT_METRIC_PROVIDER_VERSION

    def collect(
        self,
        context: EvolutionIntelligenceContext,
        findings: tuple[EvolutionAnalysisFinding, ...],
    ) -> tuple[EvolutionMetricSnapshot, ...]:
        history_ref = EvolutionIntelligenceDataSourceRef(
            source_id=f"history:{context.adaptation_id}:{context.version}",
            source_kind="evolution_history",
            description="Operation and execution history counts.",
        )
        return (
            EvolutionMetricSnapshot(
                metric_id=f"metric:operations:{context.adaptation_id}",
                metric_provider_id=self.provider_id,
                metric_provider_version=self.provider_version,
                metric_name="operation_record_count",
                metric_value=str(len(context.operation_records)),
                unit_label="count",
                data_source_refs=(history_ref,),
            ),
            EvolutionMetricSnapshot(
                metric_id=f"metric:findings:{context.adaptation_id}",
                metric_provider_id=self.provider_id,
                metric_provider_version=self.provider_version,
                metric_name="analysis_finding_count",
                metric_value=str(len(findings)),
                unit_label="count",
                data_source_refs=(history_ref,),
            ),
        )


def default_metric_providers() -> tuple[DefaultEvolutionMetricProvider,]:
    return (DefaultEvolutionMetricProvider(),)


__all__ = [
    "DefaultEvolutionMetricProvider",
    "default_metric_providers",
]
