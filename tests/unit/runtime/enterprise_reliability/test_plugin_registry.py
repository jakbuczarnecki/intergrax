# © Artur Czarnecki. All rights reserved.

"""ERL plugin architecture foundation — runtime registry and gateway tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityRiskLevel,
    EnterpriseReliabilityStrategyContext,
    ExternalEffectOutcome,
    ReconciliationStrategyAdvice,
    RiskEvaluationStrategyAdvice,
    UncertaintyLifecyclePhase,
)
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
)

pytestmark = pytest.mark.unit


def _context() -> EnterpriseReliabilityStrategyContext:
    return EnterpriseReliabilityStrategyContext(
        tenant_id="tenant-a",
        correlation_id="corr-1",
        contract_id="contract-1",
        effect_outcome=ExternalEffectOutcome.UNKNOWN,
        lifecycle_phase=UncertaintyLifecyclePhase.PENDING_RESOLUTION,
    )


@dataclass(frozen=True, slots=True)
class _ReconcilePlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor

    @property
    def plugin_id(self) -> str:
        return self._descriptor.plugin_id

    @property
    def version(self) -> str:
        return self._descriptor.version

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return self._descriptor

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> ReconciliationStrategyAdvice | None:
        if context.tenant_id != "tenant-a":
            return None
        return ReconciliationStrategyAdvice(probe_ref="status_probe")


@dataclass(frozen=True, slots=True)
class _RiskPlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor

    @property
    def plugin_id(self) -> str:
        return self._descriptor.plugin_id

    @property
    def version(self) -> str:
        return self._descriptor.version

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return self._descriptor

    def evaluate(
        self,
        context: EnterpriseReliabilityStrategyContext,
    ) -> RiskEvaluationStrategyAdvice | None:
        return RiskEvaluationStrategyAdvice(
            risk_level=EnterpriseReliabilityRiskLevel.ELEVATED,
            requires_human_review=True,
        )


def _descriptor(
    *,
    plugin_id: str,
    kind: EnterpriseReliabilityCapabilityKind,
    tenant_scope: frozenset[str] | None = None,
    priority: int = 0,
) -> EnterpriseReliabilityPluginDescriptor:
    return EnterpriseReliabilityPluginDescriptor(
        plugin_id=plugin_id,
        version="1.0.0",
        owner="test",
        capability_kind=kind,
        capabilities=("test",),
        tenant_scope=tenant_scope,
        priority=priority,
    )


def test_registry_resolves_by_capability_kind() -> None:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    reconcile = _ReconcilePlugin(
        _descriptor(
            plugin_id="reconcile-1",
            kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
        ),
    )
    registry.register(reconcile)
    assert registry.resolve_reconciliation("reconcile-1") is reconcile
    assert registry.resolve_risk_evaluation("reconcile-1") is None


def test_gateway_invokes_through_registry_port() -> None:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _ReconcilePlugin(
            _descriptor(
                plugin_id="reconcile-1",
                kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
            ),
        ),
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    advice = gateway.evaluate_reconciliation("reconcile-1", _context())
    assert advice is not None
    assert advice.probe_ref == "status_probe"
    assert gateway.evaluate_reconciliation("missing", _context()) is None


def test_list_by_capability_filters_tenant_and_orders_by_priority() -> None:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    low = _RiskPlugin(
        _descriptor(
            plugin_id="risk-low",
            kind=EnterpriseReliabilityCapabilityKind.RISK_EVALUATION,
            tenant_scope=frozenset({"tenant-a"}),
            priority=1,
        ),
    )
    high = _RiskPlugin(
        _descriptor(
            plugin_id="risk-high",
            kind=EnterpriseReliabilityCapabilityKind.RISK_EVALUATION,
            tenant_scope=None,
            priority=100,
        ),
    )
    registry.register(low)
    registry.register(high)
    listed = registry.list_by_capability(
        EnterpriseReliabilityCapabilityKind.RISK_EVALUATION,
        tenant_id="tenant-a",
    )
    assert [plugin.plugin_id for plugin in listed] == ["risk-high", "risk-low"]
    assert (
        registry.list_by_capability(
            EnterpriseReliabilityCapabilityKind.RISK_EVALUATION,
            tenant_id="tenant-z",
        )
        == (high,)
    )


def test_register_rejects_descriptor_identity_mismatch() -> None:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    descriptor = _descriptor(
        plugin_id="reconcile-1",
        kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
    )

    @dataclass(frozen=True, slots=True)
    class _MismatchedPlugin:
        def evaluate(
            self,
            context: EnterpriseReliabilityStrategyContext,
        ) -> ReconciliationStrategyAdvice | None:
            return None

        @property
        def plugin_id(self) -> str:
            return "surface-id"

        @property
        def version(self) -> str:
            return descriptor.version

        @property
        def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
            return descriptor

    with pytest.raises(ValueError, match="plugin_id mismatch"):
        registry.register(_MismatchedPlugin())
