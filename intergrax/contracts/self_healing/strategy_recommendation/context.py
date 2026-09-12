# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation engine input — quality assessments only (SELF-HEALING R5.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.quality_evaluation.assessment import StrategyQualityAssessment


@dataclass(frozen=True, slots=True)
class StrategyRecommendationCandidateQuality:
    strategy_id: str
    assessment: StrategyQualityAssessment

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if self.assessment.strategy_id != self.strategy_id:
            raise ValueError("assessment.strategy_id mismatch")


@dataclass(frozen=True, slots=True)
class StrategyRecommendationContext:
    tenant_id: str
    diagnostic_investigation_id: str
    problem_id: str
    candidates: tuple[StrategyRecommendationCandidateQuality, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.diagnostic_investigation_id.strip():
            raise ValueError("diagnostic_investigation_id required")
        if not self.problem_id.strip():
            raise ValueError("problem_id required")
        if not self.candidates:
            raise ValueError("candidates must be non-empty")
        for row in self.candidates:
            if row.assessment.tenant_id != self.tenant_id:
                raise ValueError("tenant isolation violation: assessment")


__all__ = [
    "StrategyRecommendationCandidateQuality",
    "StrategyRecommendationContext",
]
