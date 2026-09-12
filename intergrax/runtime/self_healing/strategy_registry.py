# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory self-healing strategy registry (SELF-HEALING R1)."""

from __future__ import annotations

from intergrax.contracts.self_healing.strategy import SelfHealingStrategy


class SelfHealingRegistryConfigurationError(Exception):
    """Invalid strategy registration — fail fast at bootstrap."""


class InMemorySelfHealingStrategyRegistry:
    def __init__(self) -> None:
        self._by_id: dict[str, SelfHealingStrategy] = {}

    def register(self, strategy: SelfHealingStrategy) -> None:
        descriptor = strategy.descriptor
        if descriptor.strategy_id != strategy.strategy_id:
            raise SelfHealingRegistryConfigurationError("descriptor.strategy_id mismatch")
        if descriptor.version != strategy.version:
            raise SelfHealingRegistryConfigurationError("descriptor.version mismatch")
        self._by_id[strategy.strategy_id] = strategy

    def resolve(self, strategy_id: str) -> SelfHealingStrategy | None:
        return self._by_id.get(strategy_id)

    def list_available(self, *, tenant_id: str | None = None) -> tuple[SelfHealingStrategy, ...]:
        strategies = tuple(self._by_id.values())
        if tenant_id is None:
            return strategies
        filtered: list[SelfHealingStrategy] = []
        for strategy in strategies:
            scope = strategy.descriptor.tenant_scope
            if scope is None or tenant_id in scope:
                filtered.append(strategy)
        return tuple(filtered)


__all__ = [
    "InMemorySelfHealingStrategyRegistry",
    "SelfHealingRegistryConfigurationError",
]
