# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Strategy selection SPI (SELF-HEALING R3)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.contracts.self_healing.context import SelfHealingContext
    from intergrax.contracts.self_healing.selection.performance import SelfHealingStrategyPerformance
    from intergrax.contracts.self_healing.strategy import SelfHealingStrategy


@runtime_checkable
class SelfHealingStrategySelector(Protocol):
    @property
    def selector_id(self) -> str: ...

    def select(
        self,
        available_strategies: tuple[SelfHealingStrategy, ...],
        context: SelfHealingContext,
        *,
        performance_profiles: tuple[SelfHealingStrategyPerformance, ...] = (),
    ) -> tuple[SelfHealingStrategy, ...]:
        """Return strategies in evaluation order — no execution surface."""


__all__ = ["SelfHealingStrategySelector"]
