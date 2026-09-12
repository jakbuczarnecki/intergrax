# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pluggable strategy quality evaluator port (SELF-HEALING R5.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience
from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria


@runtime_checkable
class StrategyQualityEvaluator(Protocol):
    @property
    def evaluator_id(self) -> str: ...

    def evaluate(
        self,
        criteria: StrategyQualityEvaluationCriteria,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
    ) -> StrategyQualityAssessment:
        """Analyze historical experiences — no strategy selection or execution."""
        ...


__all__ = ["StrategyQualityEvaluator"]
