# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adaptive plugin registry ports (SELF-HEALING R4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.adaptive.spi import (
    SelfHealingConfidenceEvaluator,
    SelfHealingStrategyRankingProvider,
)


@dataclass(frozen=True, slots=True)
class AdaptiveHealingPluginDescriptor:
    plugin_id: str
    version: str
    namespace: str
    priority: int
    tenant_scope: frozenset[str] | None
    timeout_seconds: float
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id required")
        if not self.version.strip():
            raise ValueError("version required")
        if not self.namespace.strip():
            raise ValueError("namespace required")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@runtime_checkable
class SelfHealingStrategyRankingRegistry(Protocol):
    def register(
        self,
        provider: SelfHealingStrategyRankingProvider,
        descriptor: AdaptiveHealingPluginDescriptor,
    ) -> None: ...

    def list_for_tenant(self, tenant_id: str) -> tuple[tuple[SelfHealingStrategyRankingProvider, AdaptiveHealingPluginDescriptor], ...]:
        ...


@runtime_checkable
class SelfHealingConfidenceEvaluatorRegistry(Protocol):
    def register(
        self,
        evaluator: SelfHealingConfidenceEvaluator,
        descriptor: AdaptiveHealingPluginDescriptor,
    ) -> None: ...

    def list_for_tenant(self, tenant_id: str) -> tuple[tuple[SelfHealingConfidenceEvaluator, AdaptiveHealingPluginDescriptor], ...]:
        ...


__all__ = [
    "AdaptiveHealingPluginDescriptor",
    "SelfHealingConfidenceEvaluatorRegistry",
    "SelfHealingStrategyRankingRegistry",
]
