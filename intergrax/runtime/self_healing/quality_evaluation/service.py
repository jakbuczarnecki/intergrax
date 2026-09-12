# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only strategy quality evaluation service (SELF-HEALING R5.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.performance_memory.query import StrategyPerformanceMemoryQuery
from intergrax.contracts.self_healing.performance_memory.repository import StrategyPerformanceMemoryRepository
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria
from intergrax.contracts.self_healing.quality_evaluation.evaluator import StrategyQualityEvaluator


@dataclass(frozen=True, slots=True)
class StrategyQualityEvaluationService:
    repository: StrategyPerformanceMemoryRepository
    evaluator: StrategyQualityEvaluator

    def assess(self, criteria: StrategyQualityEvaluationCriteria) -> StrategyQualityAssessment:
        experiences = self.repository.query(
            StrategyPerformanceMemoryQuery(
                tenant_id=criteria.tenant_id,
                strategy_id=criteria.strategy_id,
            ),
        )
        return self.evaluator.evaluate(criteria, experiences)


__all__ = ["StrategyQualityEvaluationService"]
