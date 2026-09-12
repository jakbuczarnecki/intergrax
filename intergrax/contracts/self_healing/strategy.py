# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing strategy SPI — pure decision component (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision


@dataclass(frozen=True, slots=True)
class SelfHealingStrategyDescriptor:
    strategy_id: str
    version: str
    owner: str
    capabilities: tuple[str, ...]
    tenant_scope: frozenset[str] | None
    priority: int
    specificity: int
    timeout_seconds: float
    resource_budget_tokens: int

    def __post_init__(self) -> None:
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.version.strip():
            raise ValueError("version required")
        if not self.owner.strip():
            raise ValueError("owner required")
        if not self.capabilities:
            raise ValueError("capabilities must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.resource_budget_tokens <= 0:
            raise ValueError("resource_budget_tokens must be positive")


@runtime_checkable
class SelfHealingStrategy(Protocol):
    """
    Plugin contract — evaluate only.

    Must not execute, persist Problems, emit runtime events, or bypass governance.
    """

    @property
    def strategy_id(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def descriptor(self) -> SelfHealingStrategyDescriptor: ...

    def evaluate(self, context: SelfHealingContext) -> SelfHealingDecision | None:
        """
        Return a decision when this strategy applies; ``None`` when it abstains.

        Must remain pure — no I/O.
        """


__all__ = ["SelfHealingStrategy", "SelfHealingStrategyDescriptor"]
