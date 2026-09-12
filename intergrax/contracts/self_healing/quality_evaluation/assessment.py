# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy quality assessment result — historical analysis only (SELF-HEALING R5.2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyQualityAssessment:
    """
    Descriptive quality snapshot for one strategy in one tenant scope.

    Contains aggregated history and an evaluator-produced quality score.
    No recommendations, priorities, or execution hints.
    """

    tenant_id: str
    strategy_id: str
    execution_count: int
    successful_executions: int
    failed_executions: int
    rolled_back_executions: int
    inconclusive_executions: int
    success_ratio: float
    average_recovery_time_seconds: float | None
    min_recovery_time_seconds: float | None
    max_recovery_time_seconds: float | None
    quality_score: float
    evaluator_id: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if self.execution_count < 0:
            raise ValueError("execution_count must be >= 0")
        if not self.evaluator_id.strip():
            raise ValueError("evaluator_id required")
        if not (0.0 <= self.success_ratio <= 1.0):
            raise ValueError("success_ratio must be in [0.0, 1.0]")
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError("quality_score must be in [0.0, 1.0]")
        total_outcomes = (
            self.successful_executions
            + self.failed_executions
            + self.rolled_back_executions
            + self.inconclusive_executions
        )
        if total_outcomes != self.execution_count:
            raise ValueError("outcome counts must sum to execution_count")


__all__ = ["StrategyQualityAssessment"]
