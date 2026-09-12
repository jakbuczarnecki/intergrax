# © Artur Czarnecki. All rights reserved.

"""ERL — governance and HITL integration foundation tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectEvidence,
    ExternalEffectEvidenceVerdict,
    ExternalEffectSafetyCapabilities,
    GovernanceDisposition,
    GovernanceStrategyEvaluationRequest,
    HumanApprovalRequirement,
    RecoveryDecision,
    RecoveryLifecycleAction,
    ResolutionPlatformAction,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)
from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    ExternalEffectGovernanceEvaluation,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    evaluate_external_effect_governance,
    materialize_external_effect_evidence_from_probe,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 8, 0, 0, tzinfo=UTC)
_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "governance_orchestration.py"
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


def _recovery_continue() -> RecoveryDecision:
    return RecoveryDecision(
        action=RecoveryLifecycleAction.CONTINUE,
        rationale="resume_after_erl",
    )


@dataclass(frozen=True, slots=True)
class _AllowGovernancePlugin:
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
        request: GovernanceStrategyEvaluationRequest,
    ) -> GovernanceDecision | None:
        assert request.recovery_decision.action is RecoveryLifecycleAction.CONTINUE
        assert request.resolution_decision.action is ResolutionPlatformAction.CONTINUE
        assert request.effect_contract.contract_id == "pay-1"
        assert request.execution_context.tenant_id == "tenant-a"
        return GovernanceDecision(
            disposition=GovernanceDisposition.ALLOW,
            rationale="within_autonomy",
        )


@dataclass(frozen=True, slots=True)
class _ApprovalGovernancePlugin:
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
        request: GovernanceStrategyEvaluationRequest,
    ) -> GovernanceDecision | None:
        return GovernanceDecision(
            disposition=GovernanceDisposition.APPROVAL_REQUIRED,
            rationale="high_risk_recovery",
            hitl_requirement=HumanApprovalRequirement(
                tenant_id=request.execution_context.tenant_id,
                correlation_id=request.execution_context.correlation_id,
                contract_id=request.execution_context.contract_id,
                requirement_ref="erl:governance:high_risk",
                rationale="high_risk_recovery",
            ),
        )


def _governance_descriptor() -> EnterpriseReliabilityPluginDescriptor:
    return EnterpriseReliabilityPluginDescriptor(
        plugin_id="govern-pay",
        version="1.0.0",
        owner="payments",
        capability_kind=EnterpriseReliabilityCapabilityKind.GOVERNANCE,
        capabilities=("govern",),
        tenant_scope=None,
        priority=0,
    )


def _gateway_with_governance(
    plugin: _AllowGovernancePlugin | _ApprovalGovernancePlugin,
    *,
    register: bool = True,
) -> EnterpriseReliabilityPluginGatewayImpl:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    if register:
        registry.register(plugin)
    return EnterpriseReliabilityPluginGatewayImpl(registry)


def _evaluate(
    gateway: EnterpriseReliabilityPluginGatewayImpl,
    *,
    plugin_id: str = "govern-pay",
) -> ExternalEffectGovernanceEvaluation:
    state = admit_external_effect_unknown(correlation_id="corr-pay")
    return evaluate_external_effect_governance(
        state=state,
        contract_id="pay-1",
        effect_contract=_contract(),
        resolution_decision=_resolution_continue(),
        recovery_decision=_recovery_continue(),
        evidence=_evidence(),
        gateway=gateway,
        plugin_id=plugin_id,
        tenant_id="tenant-a",
    )


def test_governance_strategy_plugin_registration_and_gateway() -> None:
    gateway = _gateway_with_governance(_AllowGovernancePlugin(_governance_descriptor()))
    evaluation = _evaluate(gateway)
    assert evaluation.governance_decision.disposition is GovernanceDisposition.ALLOW


def test_governance_allow_result_works() -> None:
    gateway = _gateway_with_governance(_AllowGovernancePlugin(_governance_descriptor()))
    evaluation = _evaluate(gateway)
    assert evaluation.governance_decision.rationale == "within_autonomy"
    assert evaluation.governance_decision.hitl_requirement is None


def test_missing_governance_strategy_fails_closed_no_allow() -> None:
    gateway = _gateway_with_governance(
        _AllowGovernancePlugin(_governance_descriptor()),
        register=False,
    )
    evaluation = _evaluate(gateway)
    assert evaluation.governance_decision.disposition is GovernanceDisposition.APPROVAL_REQUIRED
    assert evaluation.governance_decision.rationale == "governance_strategy_missing"
    assert evaluation.governance_decision.hitl_requirement is not None


def test_approval_required_preserved_with_hitl_requirement() -> None:
    gateway = _gateway_with_governance(_ApprovalGovernancePlugin(_governance_descriptor()))
    evaluation = _evaluate(gateway)
    assert evaluation.governance_decision.disposition is GovernanceDisposition.APPROVAL_REQUIRED
    requirement = evaluation.governance_decision.hitl_requirement
    assert requirement is not None
    assert requirement.requirement_ref == "erl:governance:high_risk"


def test_runtime_does_not_execute_or_self_approve() -> None:
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
    assert "evaluate_governance" in source
    assert "EnterpriseReliabilityPluginGateway" in source
