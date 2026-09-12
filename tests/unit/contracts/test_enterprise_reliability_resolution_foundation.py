# © Artur Czarnecki. All rights reserved.

"""ERL resolution strategy foundation — contract tests."""

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
    ExternalEffectOutcome,
    ExternalEffectSafetyCapabilities,
    ResolutionPlatformAction,
    ResolutionStrategy,
    ResolutionStrategyEvaluationRequest,
    UncertaintyLifecyclePhase,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    ExternalEffectEvidenceVerdict,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    materialize_external_effect_evidence_from_probe,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _StubResolutionStrategy:
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
            action=ResolutionPlatformAction.STOP,
            rationale="domain_stop",
        )


def test_resolution_strategy_protocol_is_implementable() -> None:
    plugin: ResolutionStrategy = _StubResolutionStrategy(
        EnterpriseReliabilityPluginDescriptor(
            plugin_id="resolve-1",
            version="1.0.0",
            owner="team",
            capability_kind=EnterpriseReliabilityCapabilityKind.RESOLUTION,
            capabilities=("resolve",),
            tenant_scope=None,
            priority=0,
        ),
    )
    assert isinstance(plugin, ResolutionStrategy)


def test_gateway_preserves_typed_decision() -> None:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _StubResolutionStrategy(
            EnterpriseReliabilityPluginDescriptor(
                plugin_id="resolve-1",
                version="1.0.0",
                owner="team",
                capability_kind=EnterpriseReliabilityCapabilityKind.RESOLUTION,
                capabilities=("resolve",),
                tenant_scope=None,
                priority=0,
            ),
        ),
    )
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
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("status",),
    )
    request = ResolutionStrategyEvaluationRequest(
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
    decision = gateway.evaluate_resolution("resolve-1", request)

    assert decision is not None
    assert decision.action is ResolutionPlatformAction.STOP
