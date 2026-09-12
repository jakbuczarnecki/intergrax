# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Temporal relevance for knowledge — not ML (SELF-HEALING R5.5)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@runtime_checkable
class KnowledgeFreshnessPolicy(Protocol):
    @property
    def policy_id(self) -> str: ...

    def freshness_score(
        self,
        recorded_at: datetime,
        reference_time: datetime,
    ) -> float:
        """Return weight in [0.0, 1.0] — higher means more current."""
        ...

    def order_experiences(
        self,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        reference_time: datetime,
    ) -> tuple[SelfHealingStrategyPerformanceExperience, ...]:
        """Reorder experiences for evolution input — does not drop rows."""
        ...


__all__ = ["KnowledgeFreshnessPolicy"]
