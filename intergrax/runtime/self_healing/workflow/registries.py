# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""In-memory workflow plugin registries (SELF-HEALING R2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.workflow.plan_builder import SelfHealingPlanBuilder
from intergrax.contracts.self_healing.workflow.registry import SelfHealingWorkflowPluginDescriptor
from intergrax.contracts.self_healing.workflow.rollback import SelfHealingRollbackProvider
from intergrax.contracts.self_healing.workflow.validation import SelfHealingValidationProvider


def _tenant_allowed(descriptor: SelfHealingWorkflowPluginDescriptor, tenant_id: str) -> bool:
    scope = descriptor.tenant_scope
    if scope is None:
        return True
    return tenant_id in scope


class InMemorySelfHealingPlanBuilderRegistry:
    def __init__(self) -> None:
        self._builders: dict[str, SelfHealingPlanBuilder] = {}
        self._descriptors: dict[str, SelfHealingWorkflowPluginDescriptor] = {}
        self._strategy_index: dict[str, str] = {}

    def register(
        self,
        builder: SelfHealingPlanBuilder,
        descriptor: SelfHealingWorkflowPluginDescriptor,
        *,
        strategy_id: str,
    ) -> None:
        if descriptor.plugin_id != builder.builder_id:
            raise ValueError("descriptor.plugin_id must match builder.builder_id")
        self._builders[builder.builder_id] = builder
        self._descriptors[builder.builder_id] = descriptor
        self._strategy_index[strategy_id] = builder.builder_id

    def resolve(self, builder_id: str) -> SelfHealingPlanBuilder | None:
        return self._builders.get(builder_id)

    def resolve_for_strategy(self, strategy_id: str, *, tenant_id: str) -> SelfHealingPlanBuilder | None:
        builder_id = self._strategy_index.get(strategy_id)
        if builder_id is None:
            return None
        descriptor = self._descriptors.get(builder_id)
        if descriptor is None or not _tenant_allowed(descriptor, tenant_id):
            return None
        return self._builders.get(builder_id)


class InMemorySelfHealingValidationRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SelfHealingValidationProvider] = {}
        self._descriptors: dict[str, SelfHealingWorkflowPluginDescriptor] = {}

    def register(
        self,
        provider: SelfHealingValidationProvider,
        descriptor: SelfHealingWorkflowPluginDescriptor,
    ) -> None:
        if descriptor.plugin_id != provider.provider_id:
            raise ValueError("descriptor.plugin_id must match provider.provider_id")
        self._providers[provider.provider_id] = provider
        self._descriptors[provider.provider_id] = descriptor

    def resolve(self, policy_id: str, *, tenant_id: str) -> SelfHealingValidationProvider | None:
        provider = self._providers.get(policy_id)
        if provider is None:
            return None
        descriptor = self._descriptors.get(policy_id)
        if descriptor is None or not _tenant_allowed(descriptor, tenant_id):
            return None
        return provider


class InMemorySelfHealingRollbackRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, SelfHealingRollbackProvider] = {}
        self._descriptors: dict[str, SelfHealingWorkflowPluginDescriptor] = {}

    def register(
        self,
        provider: SelfHealingRollbackProvider,
        descriptor: SelfHealingWorkflowPluginDescriptor,
    ) -> None:
        if descriptor.plugin_id != provider.provider_id:
            raise ValueError("descriptor.plugin_id must match provider.provider_id")
        self._providers[provider.provider_id] = provider
        self._descriptors[provider.provider_id] = descriptor

    def resolve(self, policy_id: str, *, tenant_id: str) -> SelfHealingRollbackProvider | None:
        provider = self._providers.get(policy_id)
        if provider is None:
            return None
        descriptor = self._descriptors.get(policy_id)
        if descriptor is None or not _tenant_allowed(descriptor, tenant_id):
            return None
        return provider


__all__ = [
    "InMemorySelfHealingPlanBuilderRegistry",
    "InMemorySelfHealingRollbackRegistry",
    "InMemorySelfHealingValidationRegistry",
]
