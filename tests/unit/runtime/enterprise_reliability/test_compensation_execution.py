# © Artur Czarnecki. All rights reserved.

"""ERL — compensation execution foundation tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pydantic import ValidationError

from intergrax.contracts.enterprise_reliability import (
    CompensationDisposition,
    CompensationExecutionError,
    CompensationExecutionOutcome,
    CompensationPlan,
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
    build_compensation_execution_request,
    build_compensation_plan,
)
from intergrax.contracts.enterprise_reliability.compensation_decision import CompensationDecision
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationPluginExecutionResult,
)
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionRequest,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityStrategyContext,
)
from intergrax.contracts.enterprise_reliability.lifecycle import UncertaintyLifecyclePhase
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    CompensationExecutionFailure,
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    execute_external_effect_compensation,
    materialize_external_effect_evidence_from_probe,
    plan_external_effect_compensation,
)
from intergrax.contracts.enterprise_reliability import (
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC)
_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "compensation_execution.py"
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


def _invoke_plan() -> object:
    resolution = _resolution_compensation_required()
    decision = CompensationDecision(
        intent=CompensationPlatformIntent.APPROVED,
        compensation_operation_ref="comp://pay/refund",
        rationale="refund_captured_payment",
    )
    return build_compensation_plan(
        resolution_decision=resolution,
        plugin_id="compensate-pay",
        decision=decision,
        strategy_registered=True,
    )


def _strategy_context() -> EnterpriseReliabilityStrategyContext:
    return EnterpriseReliabilityStrategyContext(
        tenant_id="tenant-a",
        correlation_id="corr-pay",
        contract_id="pay-1",
        effect_outcome=admit_external_effect_unknown(correlation_id="corr-pay").effect_outcome,
        lifecycle_phase=UncertaintyLifecyclePhase.PENDING_RESOLUTION,
        evidence_ref="evidence://pay/corr-pay/1",
    )


@dataclass(frozen=True, slots=True)
class _CompensationPluginWithExecutor:
    _descriptor: EnterpriseReliabilityPluginDescriptor
    _raise_on_execute: bool = False

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
        return CompensationDecision(
            intent=CompensationPlatformIntent.APPROVED,
            compensation_operation_ref="comp://pay/refund",
            rationale="refund_captured_payment",
        )

    def execute_compensation(
        self,
        request: CompensationExecutionRequest,
    ) -> CompensationPluginExecutionResult:
        if self._raise_on_execute:
            raise RuntimeError("provider_fault")
        assert request.compensation_operation_ref == "comp://pay/refund"
        return CompensationPluginExecutionResult(
            outcome=CompensationExecutionOutcome.COMPLETED,
            effect_evidence_ref="evidence://comp/corr-pay/1",
            rationale="compensation_applied",
        )


def _planning_with_gateway(
    *,
    with_executor: bool = True,
    raise_on_execute: bool = False,
) -> tuple:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    plugin = _CompensationPluginWithExecutor(
        EnterpriseReliabilityPluginDescriptor(
            plugin_id="compensate-pay",
            version="1.0.0",
            owner="payments",
            capability_kind=EnterpriseReliabilityCapabilityKind.COMPENSATION,
            capabilities=("compensate",),
            tenant_scope=None,
            priority=0,
        ),
        _raise_on_execute=raise_on_execute,
    )
    if with_executor:
        registry.register(plugin)
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
    return planning, gateway, _contract()


def test_execution_request_rejects_invalid_plan_disposition() -> None:
    plan = CompensationPlan(
        disposition=CompensationDisposition.ESCALATE_REQUIRED,
        decision=CompensationDecision(
            intent=CompensationPlatformIntent.ESCALATE,
            rationale="needs_human",
        ),
        rationale="needs_human",
    )
    assert plan.disposition is CompensationDisposition.ESCALATE_REQUIRED
    with pytest.raises(CompensationExecutionError, match="invoke_plugin"):
        build_compensation_execution_request(
            plan=plan,
            tenant_id="tenant-a",
            correlation_id="corr-pay",
            contract_id="pay-1",
            execution_context=_strategy_context(),
            effect_contract=_contract(),
        )


def test_execution_request_is_immutable_and_validates_fields() -> None:
    plan = _invoke_plan()
    request = build_compensation_execution_request(
        plan=plan,
        tenant_id="tenant-a",
        correlation_id="corr-pay",
        contract_id="pay-1",
        execution_context=_strategy_context(),
        effect_contract=_contract(),
    )
    with pytest.raises(Exception):
        request.tenant_id = "other"  # type: ignore[misc]
    payload = request.model_dump()
    payload["correlation_id"] = "mismatch"
    with pytest.raises(ValidationError):
        CompensationExecutionRequest.model_validate(payload)


def test_execute_invokes_plugin_and_returns_completed() -> None:
    planning, gateway, contract = _planning_with_gateway()
    run = execute_external_effect_compensation(
        planning=planning,
        effect_contract=contract,
        gateway=gateway,
        tenant_id="tenant-a",
    )
    assert run.execution.outcome is CompensationExecutionOutcome.COMPLETED
    assert run.execution.request is not None
    assert run.execution.plugin_result is not None
    assert run.execution.plugin_result.effect_evidence_ref == "evidence://comp/corr-pay/1"


def test_missing_executor_fails_closed_unavailable() -> None:
    @dataclass(frozen=True, slots=True)
    class _StrategyOnly:
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
            return CompensationDecision(
                intent=CompensationPlatformIntent.APPROVED,
                compensation_operation_ref="comp://pay/refund",
                rationale="ok",
            )

    state = admit_external_effect_unknown(correlation_id="corr-pay")
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(
        _StrategyOnly(
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
    run = execute_external_effect_compensation(
        planning=planning,
        effect_contract=_contract(),
        gateway=gateway,
        tenant_id="tenant-a",
    )
    assert run.execution.outcome is CompensationExecutionOutcome.UNAVAILABLE
    assert run.execution.plugin_result is None


def test_plugin_exception_produces_failed_outcome() -> None:
    planning, gateway, contract = _planning_with_gateway(raise_on_execute=True)
    run = execute_external_effect_compensation(
        planning=planning,
        effect_contract=contract,
        gateway=gateway,
        tenant_id="tenant-a",
    )
    assert run.execution.outcome is CompensationExecutionOutcome.FAILED
    assert "provider_fault" in run.execution.rationale


def test_invalid_contract_rejects_before_plugin() -> None:
    planning, gateway, _contract = _planning_with_gateway()
    bad_contract = ExternalEffectContract(
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
    with pytest.raises(CompensationExecutionFailure, match="does not declare"):
        execute_external_effect_compensation(
            planning=planning,
            effect_contract=bad_contract,
            gateway=gateway,
            tenant_id="tenant-a",
        )


def test_non_invoke_plan_does_not_call_plugin() -> None:
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
    run = execute_external_effect_compensation(
        planning=planning,
        effect_contract=_contract(),
        gateway=gateway,
        tenant_id="tenant-a",
    )
    assert run.execution.outcome is CompensationExecutionOutcome.UNAVAILABLE
    assert run.execution.request is None
    assert run.execution.plugin_result is None


def test_runtime_has_no_provider_specific_logic() -> None:
    source = _RUNTIME_SOURCE.read_text(encoding="utf-8")
    forbidden = ("sap", "refund", "stripe", "payment_api", "database_restore")
    lowered = source.lower()
    for token in forbidden:
        assert token not in lowered
    assert "execute_compensation" in source
    assert "EnterpriseReliabilityPluginGateway" in source
