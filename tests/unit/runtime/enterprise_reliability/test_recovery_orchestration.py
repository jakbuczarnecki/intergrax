# © Artur Czarnecki. All rights reserved.

"""ERL — recovery lifecycle integration foundation tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    CompensationDisposition,
    CompensationExecutionOutcome,
    CompensationPlan,
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectEvidence,
    ExternalEffectEvidenceVerdict,
    ExternalEffectSafetyCapabilities,
    RecoveryLifecycleAction,
    RecoveryStrategyEvaluationRequest,
    ResolutionPlatformAction,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    missing_compensation_strategy_decision,
)
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionResult,
)
from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    materialize_external_effect_evidence_from_probe,
    recommend_external_effect_recovery_lifecycle,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC)
_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "recovery_orchestration.py"
)


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


def _evidence() -> ExternalEffectEvidence:
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


def _resolution_continue() -> ResolutionDecision:
    return ResolutionDecision(
        action=ResolutionPlatformAction.CONTINUE,
        rationale="external_truth_confirmed",
    )


@dataclass(frozen=True, slots=True)
class _RecoveryPlugin:
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
        request: RecoveryStrategyEvaluationRequest,
    ) -> RecoveryDecision | None:
        assert request.resolution_decision.action is ResolutionPlatformAction.CONTINUE
        assert request.effect_contract.contract_id == "pay-1"
        assert request.execution_context.tenant_id == "tenant-a"
        if request.compensation_execution is not None:
            assert (
                request.compensation_execution.outcome
                is CompensationExecutionOutcome.COMPLETED
            )
        return RecoveryDecision(
            action=RecoveryLifecycleAction.CONTINUE,
            rationale="resume_after_erl",
        )


@dataclass(frozen=True, slots=True)
class _AbstainRecoveryPlugin:
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
        request: RecoveryStrategyEvaluationRequest,
    ) -> RecoveryDecision | None:
        return None


def _recovery_descriptor() -> EnterpriseReliabilityPluginDescriptor:
    return EnterpriseReliabilityPluginDescriptor(
        plugin_id="recover-pay",
        version="1.0.0",
        owner="payments",
        capability_kind=EnterpriseReliabilityCapabilityKind.RECOVERY,
        capabilities=("recover",),
        tenant_scope=None,
        priority=0,
    )


def _gateway_with_recovery(*, register: bool = True) -> EnterpriseReliabilityPluginGatewayImpl:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    if register:
        registry.register(_RecoveryPlugin(_recovery_descriptor()))
    return EnterpriseReliabilityPluginGatewayImpl(registry)


def test_recovery_strategy_plugin_registration_and_gateway() -> None:
    gateway = _gateway_with_recovery()
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    recommendation = recommend_external_effect_recovery_lifecycle(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_continue(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="recover-pay",
        tenant_id="tenant-a",
    )
    assert recommendation.recovery_decision.action is RecoveryLifecycleAction.CONTINUE


def test_erl_outcome_passed_to_strategy_including_compensation() -> None:
    gateway = _gateway_with_recovery()
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    compensation = CompensationExecutionResult(
        outcome=CompensationExecutionOutcome.COMPLETED,
        plan=CompensationPlan(
            disposition=CompensationDisposition.STRATEGY_UNAVAILABLE,
            decision=missing_compensation_strategy_decision(),
        ),
        rationale="done",
    )
    recommend_external_effect_recovery_lifecycle(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_continue(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="recover-pay",
        tenant_id="tenant-a",
        compensation_execution=compensation,
    )


def test_missing_recovery_strategy_fails_closed_no_continue() -> None:
    gateway = _gateway_with_recovery(register=False)
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    recommendation = recommend_external_effect_recovery_lifecycle(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_continue(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="recover-pay",
        tenant_id="tenant-a",
    )
    assert recommendation.recovery_decision.action is RecoveryLifecycleAction.ESCALATE
    assert recommendation.recovery_decision.rationale == "recovery_strategy_missing"


def test_abstained_recovery_strategy_waits_unresolved() -> None:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(_AbstainRecoveryPlugin(_recovery_descriptor()))
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    recommendation = recommend_external_effect_recovery_lifecycle(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_continue(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id="recover-pay",
        tenant_id="tenant-a",
    )
    assert recommendation.recovery_decision.action is RecoveryLifecycleAction.WAIT


def test_runtime_does_not_mutate_execution_lifecycle() -> None:
    source = _RUNTIME_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "ExecutionLifecyclePort",
        "apply_recovery_lifecycle_intent",
        "pause_execution",
        "resume_execution",
        "terminate_execution",
    )
    for token in forbidden:
        assert token not in source
    assert "evaluate_recovery" in source
    assert "EnterpriseReliabilityPluginGateway" in source
