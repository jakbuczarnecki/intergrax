# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Uniform freshness weights (SELF-HEALING R5.5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@dataclass(frozen=True, slots=True)
class NoDecayKnowledgeFreshnessPolicy:
    policy_id: str = "platform.no_decay_freshness"

    def freshness_score(
        self,
        recorded_at: datetime,
        reference_time: datetime,
    ) -> float:
        return 1.0

    def order_experiences(
        self,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        reference_time: datetime,
    ) -> tuple[SelfHealingStrategyPerformanceExperience, ...]:
        return tuple(sorted(experiences, key=lambda row: row.recorded_at))


__all__ = ["NoDecayKnowledgeFreshnessPolicy"]
