# © Artur Czarnecki. All rights reserved.

"""ERL — compensation strategy orchestration tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    CompensationDisposition,
    CompensationPlatformIntent,
    CompensationStrategyEvaluationRequest,
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectEvidenceVerdict,
    ExternalEffectSafetyCapabilities,
    ResolutionPlatformAction,
)
from intergrax.contracts.enterprise_reliability.compensation_decision import CompensationDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    materialize_external_effect_evidence_from_probe,
    plan_external_effect_compensation,
)
from intergrax.contracts.enterprise_reliability import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 7, 0, 0, tzinfo=UTC)


def _contract() -> ExternalEffectContract:
    return ExternalEffectContract(
        contract_id="pay-1",
        operation_key="payments.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.SUPPORTED,
        ),
        reconciliation_probe_refs=("payment_status",),
        compensation_operation_ref="comp://pay/refund",
    )


def _evidence() -> object:
    return materialize_external_effect_evidence_from_probe(
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


def _resolution_compensation_required() -> ResolutionDecision:
    return ResolutionDecision(
        action=ResolutionPlatformAction.COMPENSATION_REQUIRED,
        rationale="shipment_failed_after_capture",
    )


@dataclass(frozen=True, slots=True)
class _CompensationPlugin:
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
        request: CompensationStrategyEvaluationRequest,
    ) -> CompensationDecision | None:
        assert request.resolution_decision.action is ResolutionPlatformAction.COMPENSATION_REQUIRED
        assert request.evidence.evidence_ref == "evidence://pay/corr-pay/1"
        assert request.effect_contract.contract_id == "pay-1"
        assert request.execution_context.evidence_ref == "evidence://pay/corr-pay/1"
        return CompensationDecision(
            intent=CompensationPlatformIntent.APPROVED,
            compensation_operation_ref="comp://pay/refund",
            rationale="refund_captured_payment",
        )


def test_plan_uses_plugin_decision_when_registered() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _CompensationPlugin(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id="compensate-pay",
                version="1.0.0",
                owner="payments",
                capability_kind=EnterpriseReliabilityCapabilityKind.COMPENSATION,
                capabilities=("compensate",),
                tenant_scope=None,
                priority=0,
            ),
        ),
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    planning = plan_external_effect_compensation(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_compensation_required(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="compensate-pay",
        tenant_id="tenant-a",
    )

    assert planning.plan.disposition is CompensationDisposition.INVOKE_PLUGIN
    assert planning.plan.decision is not None
    assert planning.plan.decision.intent is CompensationPlatformIntent.APPROVED
    assert planning.plan.advice is not None
    assert planning.plan.advice.compensation_operation_ref == "comp://pay/refund"


def test_missing_strategy_fails_safely() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    gateway = EnterpriseReliabilityPluginGatewayImpl(
        InMemoryEnterpriseReliabilityPluginRegistry(),
    )
    planning = plan_external_effect_compensation(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_compensation_required(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="compensate-pay",
        tenant_id="tenant-a",
    )

    assert planning.plan.disposition is CompensationDisposition.STRATEGY_UNAVAILABLE
    assert planning.plan.decision is not None
    assert planning.plan.decision.intent is CompensationPlatformIntent.ESCALATE
    assert planning.plan.decision.intent is not CompensationPlatformIntent.APPROVED
