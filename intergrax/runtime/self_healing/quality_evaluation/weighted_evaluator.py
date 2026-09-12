# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Weighted success and recovery-time quality evaluator (SELF-HEALING R5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria
from intergrax.contracts.self_healing.quality_evaluation.statistics import (
    build_strategy_quality_assessment,
    summarize_strategy_performance_experiences,
)


def _recovery_time_component(
    average_recovery_time_seconds: float | None,
    reference_seconds: float,
) -> float:
    if average_recovery_time_seconds is None:
        return 1.0
    if reference_seconds <= 0.0:
        raise ValueError("reference_seconds must be > 0")
    ratio = average_recovery_time_seconds / reference_seconds
    return max(0.0, min(1.0, 1.0 - ratio))


@dataclass(frozen=True, slots=True)
class WeightedStrategyQualityEvaluator:
    evaluator_id: str = "platform.weighted_strategy_quality"
    recovery_time_reference_seconds: float = 60.0
    success_weight: float = 0.75
    recovery_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.recovery_time_reference_seconds <= 0.0:
            raise ValueError("recovery_time_reference_seconds must be > 0")
        weight_sum = self.success_weight + self.recovery_weight
        if abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("success_weight and recovery_weight must sum to 1.0")

    def evaluate(
        self,
        criteria: StrategyQualityEvaluationCriteria,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
    ) -> StrategyQualityAssessment:
        statistics = summarize_strategy_performance_experiences(experiences)
        if statistics.execution_count == 0:
            quality_score = 0.0
        else:
            recovery_component = _recovery_time_component(
                statistics.average_recovery_time_seconds,
                self.recovery_time_reference_seconds,
            )
            quality_score = (
                self.success_weight * statistics.success_ratio
                + self.recovery_weight * recovery_component
            )
        return build_strategy_quality_assessment(
            tenant_id=criteria.tenant_id,
            strategy_id=criteria.strategy_id,
            statistics=statistics,
            quality_score=quality_score,
            evaluator_id=self.evaluator_id,
        )


__all__ = ["WeightedStrategyQualityEvaluator"]
