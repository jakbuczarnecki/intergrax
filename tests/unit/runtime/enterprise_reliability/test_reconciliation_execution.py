# © Artur Czarnecki. All rights reserved.

"""ERL Phase 3 — reconciliation probe execution runtime tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityStrategyContext,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectEvidenceVerdict,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    ReconciliationDisposition,
    ReconciliationExecutionDisposition,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    ReconciliationStrategyAdvice,
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
    UnknownUncertaintyPosture,
)
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown_with_contract,
    execute_external_effect_reconciliation_probe,
    plan_external_effect_reconciliation,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 5, 0, 0, tzinfo=UTC)


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
        reconciliation_probe_refs=("payment_status",),
    )


@dataclass(frozen=True, slots=True)
class _ReconcilePluginWithProbe:
    _descriptor: EnterpriseReliabilityPluginDescriptor
    _verdict: ExternalEffectEvidenceVerdict = (
        ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS
    )
    _evidence_ref: str = "evidence://pay/corr-pay/1"

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
        return ReconciliationStrategyAdvice(probe_ref="payment_status")

    def execute_probe(
        self,
        request: ReconciliationProbeRequest,
    ) -> ReconciliationProbeResult:
        return ReconciliationProbeResult(
            verdict=self._verdict,
            evidence_ref=self._evidence_ref,
            rationale="provider_read_ok",
        )


def _plan_and_register(
    *,
    verdict: ExternalEffectEvidenceVerdict = ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
) -> tuple:
    contract = _payment_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-pay",
        contract=contract,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _ReconcilePluginWithProbe(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id="reconcile-pay",
                version="1.0.0",
                owner="payments",
                capability_kind=EnterpriseReliabilityCapabilityKind.RECONCILIATION,
                capabilities=("reconcile",),
                tenant_scope=None,
                priority=0,
            ),
            _verdict=verdict,
        ),
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    planning = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id="reconcile-pay",
        tenant_id="tenant-a",
    )
    return contract, planning, gateway


def test_execute_probe_resolves_success_with_evidence_fact() -> None:
    _contract, planning, gateway = _plan_and_register()
    run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id="tenant-a",
        recorded_at=_FIXED_TIME,
    )

    assert run.execution.disposition is ReconciliationExecutionDisposition.PROBE_EXECUTED
    assert run.execution.probe_result is not None
    assert run.state.effect_outcome is ExternalEffectOutcome.SUCCESS
    assert run.state.lifecycle_phase is UncertaintyLifecyclePhase.RESOLVED
    assert run.state.resolution_kind is UncertaintyResolutionKind.CONFIRMED_SUCCESS
    assert run.attempt_fact is not None
    assert run.attempt_fact.evidence_ref == "evidence://pay/corr-pay/1"
    assert run.attempt_fact.recorded_at == _FIXED_TIME


def test_execute_probe_insufficient_keeps_unknown() -> None:
    _contract, planning, gateway = _plan_and_register(
        verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT,
    )
    run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id="tenant-a",
    )

    assert run.state.effect_outcome is ExternalEffectOutcome.UNKNOWN
    assert run.state.lifecycle_phase is UncertaintyLifecyclePhase.PENDING_RESOLUTION
    assert run.attempt_fact is not None
    assert run.attempt_fact.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT


def test_execute_skipped_when_plan_not_scheduled() -> None:
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
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    planning = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id="reconcile-pay",
        tenant_id="tenant-a",
    )
    assert planning.plan.disposition is ReconciliationDisposition.ESCALATE_REQUIRED

    run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id="tenant-a",
    )
    assert (
        run.execution.disposition
        is ReconciliationExecutionDisposition.SKIPPED_NOT_SCHEDULED
    )
    assert run.attempt_fact is None


def test_execute_probe_missing_executor_on_plugin() -> None:
    contract = _payment_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id="corr-pay-2",
        contract=contract,
    )

    @dataclass(frozen=True, slots=True)
    class _PlanOnlyPlugin:
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
            return ReconciliationStrategyAdvice(probe_ref="payment_status")

    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _PlanOnlyPlugin(
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
    planning = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id="reconcile-pay",
        tenant_id="tenant-a",
    )
    run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id="tenant-a",
    )
    assert (
        run.execution.disposition
        is ReconciliationExecutionDisposition.PLUGIN_PROBE_UNAVAILABLE
    )
