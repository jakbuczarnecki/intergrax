# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing strategy registry port (SELF-HEALING R1)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.strategy import SelfHealingStrategy


@runtime_checkable
class SelfHealingStrategyRegistry(Protocol):
    def register(self, strategy: SelfHealingStrategy) -> None:
        """Register or replace a strategy by ``strategy_id``."""

    def resolve(self, strategy_id: str) -> SelfHealingStrategy | None:
        ...

    def list_available(self, *, tenant_id: str | None = None) -> tuple[SelfHealingStrategy, ...]:
        ...


__all__ = ["SelfHealingStrategyRegistry"]
