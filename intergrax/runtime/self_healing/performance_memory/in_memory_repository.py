# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory strategy performance memory (SELF-HEALING R5.1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.self_healing.performance_memory.query import StrategyPerformanceMemoryQuery
from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@dataclass
class InMemoryStrategyPerformanceMemoryRepository:
    _experiences: list[SelfHealingStrategyPerformanceExperience] = field(default_factory=list)

    def append(
        self,
        experience: SelfHealingStrategyPerformanceExperience,
    ) -> SelfHealingStrategyPerformanceExperience:
        self._experiences.append(experience)
        return experience

    def query(
        self,
        criteria: StrategyPerformanceMemoryQuery,
    ) -> tuple[SelfHealingStrategyPerformanceExperience, ...]:
        rows: list[SelfHealingStrategyPerformanceExperience] = []
        for item in self._experiences:
            if item.tenant_id != criteria.tenant_id:
                continue
            if criteria.strategy_id is not None and item.strategy_id != criteria.strategy_id:
                continue
            if criteria.workflow_id is not None and item.workflow_id != criteria.workflow_id:
                continue
            if (
                criteria.diagnostic_investigation_id is not None
                and item.diagnostic_investigation_id != criteria.diagnostic_investigation_id
            ):
                continue
            rows.append(item)
        rows.sort(key=lambda row: row.recorded_at)
        return tuple(rows)


__all__ = ["InMemoryStrategyPerformanceMemoryRepository"]
