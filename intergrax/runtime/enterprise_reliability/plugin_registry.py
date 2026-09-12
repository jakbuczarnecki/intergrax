# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory ERL plugin registry — default bootstrap implementation."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.plugin_spi import (
    CompensationExecutionStrategy,
    CompensationStrategy,
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPlugin,
    EnterpriseReliabilityPluginDescriptor,
    GovernanceStrategy,
    ReconciliationProbeExecutor,
    ReconciliationStrategy,
    RecoveryStrategy,
    ResolutionStrategy,
    RiskEvaluationStrategy,
    assert_plugin_identity_matches_descriptor,
)


class EnterpriseReliabilityPluginRegistryConfigurationError(Exception):
    """Invalid plugin registration — fail fast at bootstrap."""


def _descriptor_for(plugin: EnterpriseReliabilityPlugin) -> EnterpriseReliabilityPluginDescriptor:
    descriptor = plugin.descriptor
    assert_plugin_identity_matches_descriptor(plugin.plugin_id, plugin.version, descriptor)
    return descriptor


def _tenant_visible(
    descriptor: EnterpriseReliabilityPluginDescriptor,
    tenant_id: str | None,
) -> bool:
    if tenant_id is None:
        return True
    scope = descriptor.tenant_scope
    return scope is None or tenant_id in scope


class InMemoryEnterpriseReliabilityPluginRegistry:
    def __init__(self) -> None:
        self._reconciliation: dict[str, ReconciliationStrategy] = {}
        self._resolution: dict[str, ResolutionStrategy] = {}
        self._compensation: dict[str, CompensationStrategy] = {}
        self._recovery: dict[str, RecoveryStrategy] = {}
        self._governance: dict[str, GovernanceStrategy] = {}
        self._risk: dict[str, RiskEvaluationStrategy] = {}

    def register(self, plugin: EnterpriseReliabilityPlugin) -> None:
        descriptor = _descriptor_for(plugin)
        kind = descriptor.capability_kind
        plugin_id = plugin.plugin_id
        if kind is EnterpriseReliabilityCapabilityKind.RECONCILIATION:
            self._reconciliation[plugin_id] = plugin  # type: ignore[assignment]
            return
        if kind is EnterpriseReliabilityCapabilityKind.RESOLUTION:
            self._resolution[plugin_id] = plugin  # type: ignore[assignment]
            return
        if kind is EnterpriseReliabilityCapabilityKind.COMPENSATION:
            self._compensation[plugin_id] = plugin  # type: ignore[assignment]
            return
        if kind is EnterpriseReliabilityCapabilityKind.RECOVERY:
            self._recovery[plugin_id] = plugin  # type: ignore[assignment]
            return
        if kind is EnterpriseReliabilityCapabilityKind.GOVERNANCE:
            self._governance[plugin_id] = plugin  # type: ignore[assignment]
            return
        if kind is EnterpriseReliabilityCapabilityKind.RISK_EVALUATION:
            self._risk[plugin_id] = plugin  # type: ignore[assignment]
            return
        raise EnterpriseReliabilityPluginRegistryConfigurationError(
            f"unsupported capability kind: {kind!r}",
        )

    def resolve_reconciliation(self, plugin_id: str) -> ReconciliationStrategy | None:
        return self._reconciliation.get(plugin_id)

    def resolve_reconciliation_probe(
        self,
        plugin_id: str,
    ) -> ReconciliationProbeExecutor | None:
        plugin = self._reconciliation.get(plugin_id)
        if plugin is None:
            return None
        if isinstance(plugin, ReconciliationProbeExecutor):
            return plugin
        return None

    def resolve_resolution(self, plugin_id: str) -> ResolutionStrategy | None:
        return self._resolution.get(plugin_id)

    def resolve_compensation(self, plugin_id: str) -> CompensationStrategy | None:
        return self._compensation.get(plugin_id)

    def resolve_compensation_executor(
        self,
        plugin_id: str,
    ) -> CompensationExecutionStrategy | None:
        plugin = self._compensation.get(plugin_id)
        if plugin is None:
            return None
        if isinstance(plugin, CompensationExecutionStrategy):
            return plugin
        return None

    def resolve_recovery(self, plugin_id: str) -> RecoveryStrategy | None:
        return self._recovery.get(plugin_id)

    def resolve_governance(self, plugin_id: str) -> GovernanceStrategy | None:
        return self._governance.get(plugin_id)

    def resolve_risk_evaluation(self, plugin_id: str) -> RiskEvaluationStrategy | None:
        return self._risk.get(plugin_id)

    def list_by_capability(
        self,
        capability_kind: EnterpriseReliabilityCapabilityKind,
        *,
        tenant_id: str | None = None,
    ) -> tuple[EnterpriseReliabilityPlugin, ...]:
        if capability_kind is EnterpriseReliabilityCapabilityKind.RECONCILIATION:
            plugins = tuple(self._reconciliation.values())
        elif capability_kind is EnterpriseReliabilityCapabilityKind.RESOLUTION:
            plugins = tuple(self._resolution.values())
        elif capability_kind is EnterpriseReliabilityCapabilityKind.COMPENSATION:
            plugins = tuple(self._compensation.values())
        elif capability_kind is EnterpriseReliabilityCapabilityKind.RECOVERY:
            plugins = tuple(self._recovery.values())
        elif capability_kind is EnterpriseReliabilityCapabilityKind.GOVERNANCE:
            plugins = tuple(self._governance.values())
        elif capability_kind is EnterpriseReliabilityCapabilityKind.RISK_EVALUATION:
            plugins = tuple(self._risk.values())
        else:
            return ()
        visible = [
            plugin
            for plugin in plugins
            if _tenant_visible(_descriptor_for(plugin), tenant_id)
        ]
        return tuple(
            sorted(
                visible,
                key=lambda item: (
                    -_descriptor_for(item).priority,
                    item.plugin_id,
                ),
            ),
        )


__all__ = [
    "EnterpriseReliabilityPluginRegistryConfigurationError",
    "InMemoryEnterpriseReliabilityPluginRegistry",
]
