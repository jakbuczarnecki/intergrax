# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Simple time-weighted freshness (SELF-HEALING R5.5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@dataclass(frozen=True, slots=True)
class TimeWeightedKnowledgeFreshnessPolicy:
    """Linear decay to zero over ``half_life_seconds`` — no ML."""

    policy_id: str = "platform.time_weighted_freshness"
    half_life_seconds: float = 86_400.0 * 30.0

    def __post_init__(self) -> None:
        if self.half_life_seconds <= 0.0:
            raise ValueError("half_life_seconds must be > 0")

    def freshness_score(
        self,
        recorded_at: datetime,
        reference_time: datetime,
    ) -> float:
        age_seconds = max(0.0, (reference_time - recorded_at).total_seconds())
        if age_seconds == 0.0:
            return 1.0
        ratio = age_seconds / self.half_life_seconds
        score = max(0.0, 1.0 - ratio)
        return min(1.0, score)

    def order_experiences(
        self,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        reference_time: datetime,
    ) -> tuple[SelfHealingStrategyPerformanceExperience, ...]:
        return tuple(
            sorted(
                experiences,
                key=lambda row: (
                    self.freshness_score(row.recorded_at, reference_time),
                    row.recorded_at,
                ),
                reverse=True,
            ),
        )


__all__ = ["TimeWeightedKnowledgeFreshnessPolicy"]
