# © Artur Czarnecki. All rights reserved.

"""ERL — resolution strategy orchestration tests."""

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
    ResolutionDisposition,
    ResolutionPlatformAction,
    ResolutionStrategyEvaluationRequest,
    UncertaintyResolutionKind,
    UnknownUncertaintyPosture,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    materialize_external_effect_evidence_from_probe,
    plan_external_effect_resolution,
    ResolutionOrchestrationError,
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
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("payment_status",),
    )


def _evidence(
    *,
    verdict: ExternalEffectEvidenceVerdict = ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS,
) -> object:
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
            verdict=verdict,
            evidence_ref="evidence://pay/corr-pay/1",
        ),
        obtained_at=_FIXED_TIME,
    )


@dataclass(frozen=True, slots=True)
class _ResolutionPlugin:
    _descriptor: EnterpriseReliabilityPluginDescriptor
    _action: ResolutionPlatformAction = ResolutionPlatformAction.ESCALATE

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
        assert request.evidence.verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS
        assert request.evidence.evidence_ref == "evidence://pay/corr-pay/1"
        assert request.effect_contract.contract_id == "pay-1"
        assert request.execution_context.evidence_ref == "evidence://pay/corr-pay/1"
        return ResolutionDecision(
            action=self._action,
            rationale="human_review_before_fulfillment",
        )


def test_plan_defers_when_evidence_inconclusive() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    planning = plan_external_effect_resolution(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=_evidence(verdict=ExternalEffectEvidenceVerdict.INSUFFICIENT),
        gateway=gateway,
        plugin_id="resolve-pay",
        tenant_id="tenant-a",
    )

    assert planning.plan.disposition is ResolutionDisposition.DEFER_INSUFFICIENT_EVIDENCE


def test_plan_uses_plugin_decision_when_registered() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
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
    planning = plan_external_effect_resolution(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="resolve-pay",
        tenant_id="tenant-a",
    )

    assert planning.plan.disposition is ResolutionDisposition.INVOKE_PLUGIN
    assert planning.plan.decision is not None
    assert planning.plan.decision.action is ResolutionPlatformAction.ESCALATE
    assert planning.plan.advice is not None
    assert planning.plan.advice.resolution_kind is UncertaintyResolutionKind.ESCALATED


def test_missing_strategy_does_not_continue() -> None:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    gateway = EnterpriseReliabilityPluginGatewayImpl(
        InMemoryEnterpriseReliabilityPluginRegistry(),
    )
    planning = plan_external_effect_resolution(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="resolve-pay",
        tenant_id="tenant-a",
    )

    assert planning.plan.disposition is ResolutionDisposition.STRATEGY_UNAVAILABLE
    assert planning.plan.decision is not None
    assert planning.plan.decision.action is not ResolutionPlatformAction.CONTINUE


def test_plan_requires_unknown_outcome() -> None:
    from intergrax.runtime.enterprise_reliability import resolve_uncertainty

    state = resolve_uncertainty(
        admit_external_effect_unknown(correlation_id="corr-pay"),
        resolution_kind=UncertaintyResolutionKind.CONFIRMED_SUCCESS,
        resolved_outcome=ExternalEffectOutcome.SUCCESS,
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(
        InMemoryEnterpriseReliabilityPluginRegistry(),
    )
    with pytest.raises(ResolutionOrchestrationError):
        plan_external_effect_resolution(
            state=state,
            contract_id="pay-1",
            effect_contract=_contract(),
            unknown_posture=UnknownUncertaintyPosture.RECONCILE_ONLY,
            evidence=_evidence(),
            gateway=gateway,
            plugin_id="resolve-pay",
            tenant_id="tenant-a",
        )
