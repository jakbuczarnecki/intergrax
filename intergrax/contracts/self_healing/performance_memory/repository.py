# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy performance memory persistence port (SELF-HEALING R5.1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.performance_memory.query import StrategyPerformanceMemoryQuery
from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@runtime_checkable
class StrategyPerformanceMemoryRepository(Protocol):
    def append(
        self,
        experience: SelfHealingStrategyPerformanceExperience,
    ) -> SelfHealingStrategyPerformanceExperience: ...

    def query(
        self,
        criteria: StrategyPerformanceMemoryQuery,
    ) -> tuple[SelfHealingStrategyPerformanceExperience, ...]: ...


__all__ = ["StrategyPerformanceMemoryRepository"]
