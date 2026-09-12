# © Artur Czarnecki. All rights reserved.

"""ERL Phase 3 — reconciliation orchestration runtime tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityStrategyContext,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    ReconciliationDisposition,
    ReconciliationStrategyAdvice,
    UncertaintyLifecyclePhase,
    UnknownUncertaintyPosture,
)
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    ReconciliationOrchestrationError,
    admit_external_effect_unknown_with_contract,
    plan_external_effect_reconciliation,
)

pytestmark = pytest.mark.unit


def _payment_contract() -> ExternalEffectContract:
    return ExternalEffectContract(
        contract_id="pay-1",
        operation_key="payments.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("payment_status", "settlement_batch"),
    )


@dataclass(frozen=True, slots=True)
class _ReconcilePlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor
    _probe_ref: str = "payment_status"

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
        return ReconciliationStrategyAdvice(
            probe_ref=self._probe_ref,
            rationale="provider_status",
        )


def _register_plugin(
    registry: InMemoryEnterpriseReliabilityPluginRegistry,
    *,
    plugin_id: str = "reconcile-pay",
    probe_ref: str = "payment_status",
) -> None:
    registry.register(
        _ReconcilePlugin(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id=plugin_id,
                version="1.0.0",
                owner="payments",
                capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
                capabilities=("reconcile",),
                tenant_scope=None,
                priority=0,
            ),
            _probe_ref=probe_ref,
        ),
    )


def test_plan_uses_plugin_probe_and_advances_lifecycle() -> None:
    contract = _payment_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-pay",
        contract=contract,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    _register_plugin(registry)
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)

    result = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id="reconcile-pay",
        tenant_id="tenant-a",
    )

    assert result.plan.disposition is ReconciliationDisposition.SCHEDULE_PROBE
    assert result.plan.probe_ref == "payment_status"
    assert result.state.lifecycle_phase is UncertaintyLifecyclePhase.PENDING_RESOLUTION


def test_plan_defaults_to_first_contract_probe_when_plugin_abstains() -> None:
    contract = _payment_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-pay-2",
        contract=contract,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()

    @dataclass(frozen=True, slots=True)
    class _AbstainPlugin:
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
            return None

    registry.register(
        _AbstainPlugin(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id="reconcile-pay",
                version="1.0.0",
                owner="payments",
                capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
                capabilities=("reconcile",),
                tenant_scope=None,
                priority=0,
            ),
        ),
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)

    result = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id="reconcile-pay",
        tenant_id="tenant-a",
    )

    assert result.plan.probe_ref == "payment_status"
    assert result.plan.rationale == "platform_default_probe"


def test_plan_escalate_posture_skips_plugin_and_lifecycle() -> None:
    contract = ExternalEffectContract(
        contract_id="risky-1",
        operation_key="legacy.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-risk",
        contract=contract,
    )
    assert admission.unknown_posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    _register_plugin(registry)
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)

    result = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id="reconcile-pay",
        tenant_id="tenant-a",
    )

    assert result.plan.disposition is ReconciliationDisposition.ESCALATE_REQUIRED
    assert result.state.lifecycle_phase is UncertaintyLifecyclePhase.ADMITTED


def test_plan_rejects_undeclared_plugin_probe() -> None:
    contract = _payment_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-pay-3",
        contract=contract,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    _register_plugin(registry, probe_ref="undeclared_probe")
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)

    with pytest.raises(ReconciliationOrchestrationError, match="not declared"):
        plan_external_effect_reconciliation(
            admission=admission,
            contract=contract,
            gateway=gateway,
            plugin_id="reconcile-pay",
            tenant_id="tenant-a",
        )
