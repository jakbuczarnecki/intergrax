# © Artur Czarnecki. All rights reserved.

"""ERL compensation strategy foundation — contract tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from intergrax.contracts.enterprise_reliability import (
    CompensationPlatformIntent,
    CompensationStrategy,
    CompensationStrategyEvaluationRequest,
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityStrategyContext,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectEvidenceVerdict,
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    ResolutionPlatformAction,
    UncertaintyLifecyclePhase,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)
from intergrax.contracts.enterprise_reliability.compensation_decision import CompensationDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    materialize_external_effect_evidence_from_probe,
)

pytestmark = pytest.mark.unit


def _descriptor() -> EnterpriseReliabilityPluginDescriptor:
    return EnterpriseReliabilityPluginDescriptor(
        plugin_id="compensate-1",
        version="1.0.0",
        owner="team",
        capability_kind=EnterpriseReliabilityCapabilityKind.COMPENSATION,
        capabilities=("compensate",),
        tenant_scope=None,
        priority=0,
    )


def _stub_compensation_strategy(
    seen_resolutions: list[ResolutionDecision],
) -> CompensationStrategy:
    descriptor = _descriptor()

    @dataclass(frozen=True, slots=True)
    class _Stub:
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
            seen_resolutions.append(request.resolution_decision)
            return CompensationDecision(
                intent=CompensationPlatformIntent.APPROVED,
                compensation_operation_ref="comp://ops/refund",
                rationale="offset_prior_effect",
            )

    return _Stub(descriptor)


def test_compensation_strategy_protocol_is_implementable() -> None:
    plugin = _stub_compensation_strategy([])
    assert isinstance(plugin, CompensationStrategy)


def test_gateway_preserves_typed_compensation_decision() -> None:
    seen: list[ResolutionDecision] = []
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(_stub_compensation_strategy(seen))
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    evidence = materialize_external_effect_evidence_from_probe(
        probe_request=ReconciliationProbeRequest(
            tenant_id="t1",
            correlation_id="c1",
            contract_id="contract-1",
            probe_ref="status",
            plugin_id="reconcile-1",
            attempt_index=1,
        ),
        probe_result=ReconciliationProbeResult(
            verdict=ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE,
            evidence_ref="evidence://c1/1",
        ),
        obtained_at=datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC),
    )
    contract = ExternalEffectContract(
        contract_id="contract-1",
        operation_key="ops.charge",
        category=ExternalEffectCategory.FINANCIAL,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.SUPPORTED,
        ),
        reconciliation_probe_refs=("status",),
        compensation_operation_ref="comp://contract/refund",
    )
    resolution = ResolutionDecision(
        action=ResolutionPlatformAction.COMPENSATION_REQUIRED,
        rationale="downstream_failure",
    )
    request = CompensationStrategyEvaluationRequest(
        resolution_decision=resolution,
        evidence=evidence,
        execution_context=EnterpriseReliabilityStrategyContext(
            tenant_id="t1",
            correlation_id="c1",
            contract_id="contract-1",
            effect_outcome=ExternalEffectOutcome.UNKNOWN,
            lifecycle_phase=UncertaintyLifecyclePhase.PENDING_RESOLUTION,
            evidence_verdict=evidence.verdict,
            evidence_ref=evidence.evidence_ref,
        ),
        effect_contract=contract,
    )
    decision = gateway.evaluate_compensation("compensate-1", request)

    assert decision is not None
    assert decision.intent is CompensationPlatformIntent.APPROVED
    assert seen == [resolution]
    assert seen[0].action is ResolutionPlatformAction.COMPENSATION_REQUIRED
