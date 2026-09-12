# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default strategy selectors (SELF-HEALING R3)."""

from __future__ import annotations

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.selection.performance import SelfHealingStrategyPerformance
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy
from intergrax.runtime.self_healing.resolution import resolve_strategies_for_context


class HighestConfidenceStrategySelector:
    selector_id = "platform.highest_confidence"

    def select(
        self,
        available_strategies: tuple[SelfHealingStrategy, ...],
        context: SelfHealingContext,
        *,
        performance_profiles: tuple[SelfHealingStrategyPerformance, ...] = (),
    ) -> tuple[SelfHealingStrategy, ...]:
        perf_map = {p.strategy_id: p for p in performance_profiles}
        resolved = resolve_strategies_for_context(available_strategies, context)

        def sort_key(strategy: SelfHealingStrategy) -> tuple[float, float, int, str]:
            perf = perf_map.get(strategy.strategy_id)
            success = perf.success_rate if perf is not None else 0.5
            calibration = perf.confidence_calibration if perf is not None else 0.5
            return (success, calibration, strategy.descriptor.priority, strategy.strategy_id)

        return tuple(sorted(resolved, key=sort_key, reverse=True))


__all__ = ["HighestConfidenceStrategySelector"]
