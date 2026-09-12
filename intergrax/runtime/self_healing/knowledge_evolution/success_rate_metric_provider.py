# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Success-rate metric provider (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.metrics import (
    StrategyMetricBundle,
    StrategyMetricScope,
    StrategyMetricValue,
)
from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience
from intergrax.contracts.self_healing.quality_evaluation.statistics import summarize_strategy_performance_experiences


@dataclass(frozen=True, slots=True)
class SuccessRateMetricProvider:
    provider_id: str = "platform.success_rate_metric"

    def collect(
        self,
        scope: StrategyMetricScope,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
    ) -> StrategyMetricBundle:
        statistics = summarize_strategy_performance_experiences(experiences)
        metrics: tuple[StrategyMetricValue, ...] = (
            StrategyMetricValue(
                name="success_rate",
                value=statistics.success_ratio,
                unit="ratio",
                evidence_refs=statistics.evidence_refs,
            ),
            StrategyMetricValue(
                name="failure_rate",
                value=1.0 - statistics.success_ratio if statistics.execution_count > 0 else 0.0,
                unit="ratio",
                evidence_refs=statistics.evidence_refs,
            ),
        )
        return StrategyMetricBundle(
            provider_id=self.provider_id,
            scope=scope,
            metrics=metrics,
        )


__all__ = ["SuccessRateMetricProvider"]
