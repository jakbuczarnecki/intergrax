# © Artur Czarnecki. All rights reserved.

"""ERL — resolution strategy execution tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
    ExternalEffectOutcome,
    ResolutionExecutionDisposition,
    ResolutionPlatformAction,
    ResolutionStrategyEvaluationRequest,
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
    UnknownUncertaintyPosture,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    ExternalEffectEvidenceVerdict,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    execute_external_effect_resolution,
    materialize_external_effect_evidence_from_probe,
    plan_external_effect_resolution,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 7, 30, 0, tzinfo=UTC)


def _contract() -> ExternalEffectContract:
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
class _ResolutionPlugin:
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
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None:
        return ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="continue_fulfillment",
        )


def _planning_with_plugin() -> object:
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=ReconciliationProbeRequest(
            tenant_id="tenant-a",
            correlation_id="corr-pay",
            contract_id="pay-1",
            probe_ref="payment_status",
            plugin_id="reconcile-pay",
            attempt_index=1,
        ),
        probe_result=ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref="evidence://pay/corr-pay/1",
        ),
        obtained_at=_FIXED_TIME,
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _ResolutionPlugin(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id="resolve-pay",
                version="1.0.0",
                owner="payments",
                capability_kind=EnterpriseReliabilityCapabilityKind.RESOLUTION,
                capabilities=("resolve",),
                tenant_scope=None,
                priority=0,
            ),
        ),
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    return plan_external_effect_resolution(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=evidence,
        gateway=gateway,
        plugin_id="resolve-pay",
        tenant_id="tenant-a",
    )


def test_execute_applies_plugin_advice_and_emits_fact() -> None:
    planning = _planning_with_plugin()
    run = execute_external_effect_resolution(planning=planning, recorded_at=_FIXED_TIME)

    assert run.execution.disposition is ResolutionExecutionDisposition.ADVICE_APPLIED
    assert run.state.effect_outcome is ExternalEffectOutcome.SUCCESS
    assert run.state.lifecycle_phase is UncertaintyLifecyclePhase.RESOLVED
    assert run.decision_fact is not None
    assert run.decision_fact.plugin_id == "resolve-pay"
    assert run.decision_fact.platform_action is ResolutionPlatformAction.CONTINUE
    assert run.decision_fact.recorded_at == _FIXED_TIME


def test_execute_skips_when_resolution_strategy_missing() -> None:
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=ReconciliationProbeRequest(
            tenant_id="tenant-a",
            correlation_id="corr-pay",
            contract_id="pay-1",
            probe_ref="payment_status",
            plugin_id="reconcile-pay",
            attempt_index=1,
        ),
        probe_result=ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
            evidence_ref="evidence://pay/corr-pay/1",
        ),
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(
        InMemoryEnterpriseReliabilityPluginRegistry(),
    )
    planning = plan_external_effect_resolution(
        state=admit_external_effect_unknown(correlation_id="corr-pay"),
        contract_id="pay-1",
        effect_contract=_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=evidence,
        gateway=gateway,
        plugin_id="resolve-pay",
        tenant_id="tenant-a",
    )
    run = execute_external_effect_resolution(planning=planning)

    assert run.execution.disposition is ResolutionExecutionDisposition.SKIPPED_PLUGIN_UNAVAILABLE
    assert run.state.effect_outcome is ExternalEffectOutcome.UNKNOWN
    assert run.decision_fact is None
