# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing workflow plugin descriptors and registry ports (SELF-HEALING R2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.workflow.plan_builder import SelfHealingPlanBuilder
from intergrax.contracts.self_healing.workflow.rollback import SelfHealingRollbackProvider
from intergrax.contracts.self_healing.workflow.validation import SelfHealingValidationProvider
from intergrax.contracts.self_healing.registry import SelfHealingStrategyRegistry


@dataclass(frozen=True, slots=True)
class SelfHealingWorkflowPluginDescriptor:
    plugin_id: str
    version: str
    namespace: str
    priority: int
    capabilities: tuple[str, ...]
    tenant_scope: frozenset[str] | None
    timeout_seconds: float

    def __post_init__(self) -> None:
        if not self.plugin_id.strip():
            raise ValueError("plugin_id required")
        if not self.version.strip():
            raise ValueError("version required")
        if not self.namespace.strip():
            raise ValueError("namespace required")
        if not self.capabilities:
            raise ValueError("capabilities must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@runtime_checkable
class SelfHealingPlanBuilderRegistry(Protocol):
    def register(self, builder: SelfHealingPlanBuilder, descriptor: SelfHealingWorkflowPluginDescriptor) -> None:
        ...

    def resolve(self, builder_id: str) -> SelfHealingPlanBuilder | None:
        ...

    def resolve_for_strategy(self, strategy_id: str, *, tenant_id: str) -> SelfHealingPlanBuilder | None:
        ...


@runtime_checkable
class SelfHealingValidationRegistry(Protocol):
    def register(
        self,
        provider: SelfHealingValidationProvider,
        descriptor: SelfHealingWorkflowPluginDescriptor,
    ) -> None:
        ...

    def resolve(self, policy_id: str, *, tenant_id: str) -> SelfHealingValidationProvider | None:
        ...


@runtime_checkable
class SelfHealingRollbackRegistry(Protocol):
    def register(
        self,
        provider: SelfHealingRollbackProvider,
        descriptor: SelfHealingWorkflowPluginDescriptor,
    ) -> None:
        ...

    def resolve(self, policy_id: str, *, tenant_id: str) -> SelfHealingRollbackProvider | None:
        ...


__all__ = [
    "SelfHealingPlanBuilderRegistry",
    "SelfHealingRollbackRegistry",
    "SelfHealingStrategyRegistry",
    "SelfHealingValidationRegistry",
    "SelfHealingWorkflowPluginDescriptor",
]
