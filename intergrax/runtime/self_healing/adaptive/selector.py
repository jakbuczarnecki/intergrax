# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive strategy selector — extends R3 selection without replacing lifecycle (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.selection.performance import SelfHealingStrategyPerformance
from intergrax.contracts.self_healing.selection.selector import SelfHealingStrategySelector
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy
from intergrax.runtime.self_healing.adaptive.engine import AdaptiveSelfHealingEngine
from intergrax.runtime.self_healing.lifecycle.selection import HighestConfidenceStrategySelector


class AdaptiveStrategySelector:
    """
    Wraps default selector with adaptive ranking recommendation.

    Does not execute healing or alter lifecycle engine authority.
    """

    selector_id = "platform.adaptive.strategy_selector"

    def __init__(
        self,
        adaptive_engine: AdaptiveSelfHealingEngine,
        *,
        fallback_selector: SelfHealingStrategySelector | None = None,
    ) -> None:
        self._engine = adaptive_engine
        self._fallback = fallback_selector or HighestConfidenceStrategySelector()

    def select(
        self,
        available_strategies: tuple[SelfHealingStrategy, ...],
        context: SelfHealingContext,
        *,
        performance_profiles: tuple[SelfHealingStrategyPerformance, ...] = (),
    ) -> tuple[SelfHealingStrategy, ...]:
        adaptive_context = self._engine.build_context(context, available_strategies)
        recommendation = self._engine.recommend(adaptive_context)
        by_id = {s.strategy_id: s for s in available_strategies}
        ordered: list[SelfHealingStrategy] = []
        seen: set[str] = set()
        for strategy_id in recommendation.recommended_strategy_order:
            strategy = by_id.get(strategy_id)
            if strategy is not None and strategy_id not in seen:
                ordered.append(strategy)
                seen.add(strategy_id)
        fallback_order = self._fallback.select(
            available_strategies,
            context,
            performance_profiles=performance_profiles,
        )
        for strategy in fallback_order:
            if strategy.strategy_id not in seen:
                ordered.append(strategy)
                seen.add(strategy.strategy_id)
        return tuple(ordered)


__all__ = ["AdaptiveStrategySelector"]
