# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pluggable strategy learning engine port (SELF-HEALING R5.4)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.comparison import StrategyComparisonResult
from intergrax.contracts.self_healing.knowledge_evolution.evolution import (
    StrategyKnowledgeEvolutionContext,
    StrategyKnowledgeEvolutionResult,
)
from intergrax.contracts.self_healing.knowledge_evolution.metrics import StrategyMetricBundle
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeProfile
from intergrax.contracts.self_healing.performance_memory.record import SelfHealingStrategyPerformanceExperience


@runtime_checkable
class StrategyLearningEngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    def evolve(
        self,
        context: StrategyKnowledgeEvolutionContext,
        current_profile: StrategyKnowledgeProfile | None,
        experiences: tuple[SelfHealingStrategyPerformanceExperience, ...],
        metrics: StrategyMetricBundle,
        comparison: StrategyComparisonResult | None,
    ) -> StrategyKnowledgeEvolutionResult:
        """Produce next profile revision — no execution or selection authority."""
        ...


__all__ = ["StrategyLearningEngine"]
