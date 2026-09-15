# © Artur Czarnecki. All rights reserved.

"""Composable recall decision strategies (MEM-ENT-6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.strategies.defaults.conservative_conflict import (
    ConservativeMemoryConflictDetectionStrategy,
    FailSafeMemoryConflictResolutionStrategy,
)
from intergrax.memory.strategies.defaults.enterprise_ranking import EnterpriseMemoryRankingStrategy
from intergrax.memory.strategies.protocols import (
    MemoryConflictDetectionStrategy,
    MemoryConflictResolutionStrategy,
    MemoryRankingStrategy,
)


@dataclass(frozen=True, slots=True)
class MemoryRecallStrategySet:
    ranking: MemoryRankingStrategy
    conflict_detection: MemoryConflictDetectionStrategy
    conflict_resolution: MemoryConflictResolutionStrategy


def build_default_memory_recall_strategies() -> MemoryRecallStrategySet:
    return MemoryRecallStrategySet(
        ranking=EnterpriseMemoryRankingStrategy(),
        conflict_detection=ConservativeMemoryConflictDetectionStrategy(),
        conflict_resolution=FailSafeMemoryConflictResolutionStrategy(),
    )
