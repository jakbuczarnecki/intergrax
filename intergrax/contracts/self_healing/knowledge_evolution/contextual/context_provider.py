# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Pluggable strategy context resolution (SELF-HEALING R5.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.knowledge_evolution.contextual.operating_context import (
    StrategyKnowledgeOperatingContext,
)
from intergrax.contracts.self_healing.knowledge_evolution.profile import StrategyKnowledgeContext


@dataclass(frozen=True, slots=True)
class StrategyContextResolutionRequest:
    tenant_id: str
    strategy_id: str
    evolution_scope: StrategyKnowledgeContext
    trigger_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")
        if not self.strategy_id.strip():
            raise ValueError("strategy_id required")
        if not self.trigger_refs:
            raise ValueError("trigger_refs must be non-empty")


@runtime_checkable
class StrategyContextProvider(Protocol):
    @property
    def provider_id(self) -> str: ...

    def resolve(
        self,
        request: StrategyContextResolutionRequest,
    ) -> StrategyKnowledgeOperatingContext | None:
        """Supply descriptive context — no execution or selection authority."""
        ...


__all__ = ["StrategyContextProvider", "StrategyContextResolutionRequest"]
