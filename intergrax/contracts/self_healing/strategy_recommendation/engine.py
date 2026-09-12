# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pluggable strategy recommendation engine port (SELF-HEALING R5.3)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.strategy_recommendation.context import StrategyRecommendationContext
from intergrax.contracts.self_healing.strategy_recommendation.recommendation import StrategyRecommendation


@runtime_checkable
class StrategyRecommendationEngine(Protocol):
    @property
    def engine_id(self) -> str: ...

    def recommend(self, context: StrategyRecommendationContext) -> StrategyRecommendation:
        """Produce an advisory recommendation — no strategy execution or selection authority."""
        ...


__all__ = ["StrategyRecommendationEngine"]
