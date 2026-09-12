# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory adaptive plugin registries (SELF-HEALING R4)."""

from __future__ import annotations

from intergrax.contracts.self_healing.adaptive.registry import AdaptiveHealingPluginDescriptor
from intergrax.contracts.self_healing.adaptive.spi import (
    SelfHealingConfidenceEvaluator,
    SelfHealingStrategyRankingProvider,
)


def _tenant_allowed(descriptor: AdaptiveHealingPluginDescriptor, tenant_id: str) -> bool:
    scope = descriptor.tenant_scope
    if scope is None:
        return True
    return tenant_id in scope


class InMemorySelfHealingStrategyRankingRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SelfHealingStrategyRankingProvider] = {}
        self._descriptors: dict[str, AdaptiveHealingPluginDescriptor] = {}

    def register(
        self,
        provider: SelfHealingStrategyRankingProvider,
        descriptor: AdaptiveHealingPluginDescriptor,
    ) -> None:
        if descriptor.plugin_id != provider.provider_id:
            raise ValueError("descriptor.plugin_id must match provider.provider_id")
        self._providers[provider.provider_id] = provider
        self._descriptors[provider.provider_id] = descriptor

    def list_for_tenant(
        self,
        tenant_id: str,
    ) -> tuple[tuple[SelfHealingStrategyRankingProvider, AdaptiveHealingPluginDescriptor], ...]:
        entries: list[tuple[SelfHealingStrategyRankingProvider, AdaptiveHealingPluginDescriptor, int]] = []
        for plugin_id, provider in self._providers.items():
            descriptor = self._descriptors.get(plugin_id)
            if descriptor is None or not _tenant_allowed(descriptor, tenant_id):
                continue
            entries.append((provider, descriptor, descriptor.priority))
        entries.sort(key=lambda item: item[2], reverse=True)
        return tuple((provider, descriptor) for provider, descriptor, _ in entries)


class InMemorySelfHealingConfidenceEvaluatorRegistry:
    def __init__(self) -> None:
        self._evaluators: dict[str, SelfHealingConfidenceEvaluator] = {}
        self._descriptors: dict[str, AdaptiveHealingPluginDescriptor] = {}

    def register(
        self,
        evaluator: SelfHealingConfidenceEvaluator,
        descriptor: AdaptiveHealingPluginDescriptor,
    ) -> None:
        if descriptor.plugin_id != evaluator.evaluator_id:
            raise ValueError("descriptor.plugin_id must match evaluator.evaluator_id")
        self._evaluators[evaluator.evaluator_id] = evaluator
        self._descriptors[evaluator.evaluator_id] = descriptor

    def list_for_tenant(
        self,
        tenant_id: str,
    ) -> tuple[tuple[SelfHealingConfidenceEvaluator, AdaptiveHealingPluginDescriptor], ...]:
        entries: list[tuple[SelfHealingConfidenceEvaluator, AdaptiveHealingPluginDescriptor, int]] = []
        for plugin_id, evaluator in self._evaluators.items():
            descriptor = self._descriptors.get(plugin_id)
            if descriptor is None or not _tenant_allowed(descriptor, tenant_id):
                continue
            entries.append((evaluator, descriptor, descriptor.priority))
        entries.sort(key=lambda item: item[2], reverse=True)
        return tuple((evaluator, descriptor) for evaluator, descriptor, _ in entries)


__all__ = [
    "InMemorySelfHealingConfidenceEvaluatorRegistry",
    "InMemorySelfHealingStrategyRankingRegistry",
]
