# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy quality assessment port — repository-backed read model (SELF-HEALING R5.2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment
from intergrax.contracts.self_healing.quality_evaluation.criteria import StrategyQualityEvaluationCriteria


@runtime_checkable
class StrategyQualityAssessor(Protocol):
    def assess(self, criteria: StrategyQualityEvaluationCriteria) -> StrategyQualityAssessment:
        ...


__all__ = ["StrategyQualityAssessor"]
