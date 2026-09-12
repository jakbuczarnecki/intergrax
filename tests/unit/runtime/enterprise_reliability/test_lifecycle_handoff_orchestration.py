# © Artur Czarnecki. All rights reserved.

"""ERL — recovery lifecycle execution handoff foundation tests."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    LifecycleHandoffDisposition,
    RecoveryDecision,
    RecoveryLifecycleAction,
    RecoveryLifecycleIntent,
    ResolutionPlatformAction,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
)
from intergrax.contracts.enterprise_reliability.execution_lifecycle_port import (
    ExecutionLifecyclePort,
)
from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    ExternalEffectGovernanceEvaluation,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    evaluate_external_effect_governance,
    handoff_recovery_lifecycle_to_execution,
    materialize_external_effect_evidence_from_probe,
)

pytestmark = pytest.mark.unit

_FIXED_TIME = datetime(2026, 9, 12, 9, 0, 0, tzinfo=UTC)
_RUNTIME_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "enterprise_reliability"
    / "lifecycle_handoff_orchestration.py"
)
_EXECUTION_REF = "execution://corr-pay/run-1"


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
        return GovernanceDecision(
            disposition=GovernanceDisposition.ALLOW,
            rationale="within_autonomy",
        )


@dataclass(frozen=True, slots=True)
class _DenyGovernancePlugin:
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
            disposition=GovernanceDisposition.DENY,
            rationale="policy_deny",
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
            rationale="needs_human",
            hitl_requirement=HumanApprovalRequirement(
                tenant_id=request.execution_context.tenant_id,
                correlation_id=request.execution_context.correlation_id,
                contract_id=request.execution_context.contract_id,
                requirement_ref="erl:governance:test",
                rationale="needs_human",
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


def _gateway_with_plugin(
    plugin: _AllowGovernancePlugin | _DenyGovernancePlugin | _ApprovalGovernancePlugin,
) -> EnterpriseReliabilityPluginGatewayImpl:
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    registry.register(plugin)
    return EnterpriseReliabilityPluginGatewayImpl(registry)


def _governance_evaluation(
    gateway: EnterpriseReliabilityPluginGatewayImpl,
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
        plugin_id="govern-pay",
        tenant_id="tenant-a",
    )


@dataclass
class _RecordingLifecyclePort:
    intents: list[RecoveryLifecycleIntent] = field(default_factory=list)

    def apply_recovery_lifecycle_intent(self, intent: RecoveryLifecycleIntent) -> None:
        self.intents.append(intent)


def test_approved_governance_creates_lifecycle_handoff() -> None:
    gateway = _gateway_with_plugin(_AllowGovernancePlugin(_governance_descriptor()))
    evaluation = _governance_evaluation(gateway)
    port = _RecordingLifecyclePort()
    result = handoff_recovery_lifecycle_to_execution(
        governance_evaluation=evaluation,
        execution_ref=_EXECUTION_REF,
        tenant_id="tenant-a",
        lifecycle_port=port,
    )
    assert result.disposition is LifecycleHandoffDisposition.HANDED_OFF
    assert result.request is not None
    assert result.request.execution_ref == _EXECUTION_REF
    assert result.request.lifecycle_action is RecoveryLifecycleAction.CONTINUE
    assert result.request.governance_result_ref is not None
    assert len(port.intents) == 1
    assert port.intents[0].correlation_id == "corr-pay"
    assert port.intents[0].decision.action is RecoveryLifecycleAction.CONTINUE


def test_rejected_governance_blocks_handoff() -> None:
    gateway = _gateway_with_plugin(_DenyGovernancePlugin(_governance_descriptor()))
    evaluation = _governance_evaluation(gateway)
    port = _RecordingLifecyclePort()
    result = handoff_recovery_lifecycle_to_execution(
        governance_evaluation=evaluation,
        execution_ref=_EXECUTION_REF,
        tenant_id="tenant-a",
        lifecycle_port=port,
    )
    assert result.disposition is LifecycleHandoffDisposition.BLOCKED
    assert len(port.intents) == 0


def test_approval_required_blocks_execution_handoff() -> None:
    gateway = _gateway_with_plugin(_ApprovalGovernancePlugin(_governance_descriptor()))
    evaluation = _governance_evaluation(gateway)
    port = _RecordingLifecyclePort()
    result = handoff_recovery_lifecycle_to_execution(
        governance_evaluation=evaluation,
        execution_ref=_EXECUTION_REF,
        tenant_id="tenant-a",
        lifecycle_port=port,
    )
    assert result.disposition is LifecycleHandoffDisposition.APPROVAL_REQUIRED
    assert len(port.intents) == 0


def test_execution_lifecycle_port_invoked_through_abstraction() -> None:
    gateway = _gateway_with_plugin(_AllowGovernancePlugin(_governance_descriptor()))
    evaluation = _governance_evaluation(gateway)
    port = _RecordingLifecyclePort()
    handoff_recovery_lifecycle_to_execution(
        governance_evaluation=evaluation,
        execution_ref=_EXECUTION_REF,
        tenant_id="tenant-a",
        lifecycle_port=port,
    )
    assert isinstance(port, ExecutionLifecyclePort)


def test_missing_lifecycle_port_returns_typed_failure() -> None:
    gateway = _gateway_with_plugin(_AllowGovernancePlugin(_governance_descriptor()))
    evaluation = _governance_evaluation(gateway)
    result = handoff_recovery_lifecycle_to_execution(
        governance_evaluation=evaluation,
        execution_ref=_EXECUTION_REF,
        tenant_id="tenant-a",
        lifecycle_port=None,
    )
    assert result.disposition is LifecycleHandoffDisposition.PORT_UNAVAILABLE
    assert result.rationale == "execution_lifecycle_port_unavailable"


def test_runtime_does_not_mutate_execution_state_directly() -> None:
    source = _RUNTIME_SOURCE.read_text(encoding="utf-8")
    forbidden = (
        "pause_execution",
        "resume_execution",
        "terminate_execution",
    )
    for token in forbidden:
        assert token not in source
    assert "apply_recovery_lifecycle_intent" in source
    assert "ExecutionLifecyclePort" in source
