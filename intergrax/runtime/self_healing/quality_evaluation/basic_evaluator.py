# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Success-ratio strategy quality evaluator (SELF-HEALING R5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria
from intergrax.contracts.self_healing.quality_evaluation.statistics import (
    build_strategy_quality_assessment,
    summarize_strategy_performance_experiences,
)


@dataclass(frozen=True, slots=True)
class BasicStrategyQualityEvaluator:
    evaluator_id: str = "platform.basic_strategy_quality"

    def evaluate(
        self,
        criteria: StrategyQualityEvaluationCriteria,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
    ) -> StrategyQualityAssessment:
        statistics = summarize_strategy_performance_experiences(experiences)
        quality_score = statistics.success_ratio if statistics.execution_count > 0 else 0.0
        return build_strategy_quality_assessment(
            tenant_id=criteria.tenant_id,
            strategy_id=criteria.strategy_id,
            statistics=statistics,
            quality_score=quality_score,
            evaluator_id=self.evaluator_id,
        )


__all__ = ["BasicStrategyQualityEvaluator"]
