# © Artur Czarnecki. All rights reserved.

"""GR-7-A7 — controlled provider invocation recovery decision and execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from applications.governed_contractor_application.host.provider_invocation_recovery import (
    GovernedExternalWorkProviderRecovery,
)
from applications.governed_contractor_application.host.provider_invocation_reconciliation import (
    GovernedExternalWorkProviderReconciliation,
)
from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.side_effect_actions import ACTION_CREATE_EXTERNAL_WORK
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectCapabilitySupport,
    ExternalEffectCategory,
    ExternalEffectContract,
    ExternalEffectSafetyCapabilities,
    UnknownUncertaintyPosture,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationReason,
    ProviderInvocationReconciliationResult,
    ProviderInvocationReconciliationVerdict,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryPolicyDecision,
    ProviderInvocationRecoveryPolicyRequest,
    ProviderInvocationRecoveryReason,
    ProviderInvocationRecoveryRequest,
    evaluate_provider_invocation_recovery,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityReason,
    ExternalEffectRepeatEligibilityRequest,
    ExternalEffectRepeatEligibilityResult,
    ExternalEffectRepeatEligibilityVerdict,
    ExternalEffectRepeatPolicyDecision,
    ExternalEffectRepeatPolicyRequest,
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.runtime.enterprise_reliability.default_provider_invocation_recovery_policy import (
    default_fail_closed_provider_invocation_recovery_policy,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionDisposition,
    ProviderInvocationRecoveryExecutionPorts,
    ProviderInvocationRecoveryHitlResult,
    ProviderInvocationRecoveryRepeatResult,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 17, 10, 0, 0, tzinfo=UTC)
_TENANT = "gr7a7-tenant"
_CAPABILITIES = quote_first_partner_capability_fixture()
_CREATE_CONTRACT = external_work_effect_contract_for_action(
    ACTION_CREATE_EXTERNAL_WORK,
    _CAPABILITIES,
)


@dataclass(frozen=True, slots=True)
class _AllowRepeatPolicy:
    def decide(self, request: ExternalEffectRepeatPolicyRequest) -> ExternalEffectRepeatPolicyDecision:
        _ = request
        return ExternalEffectRepeatPolicyDecision(allow_repeat=True)

    @property
    def policy_id(self) -> str:
        return "test_allow_repeat"


@dataclass(frozen=True, slots=True)
class _SelectIdempotentRepeatRecoveryPolicy:
    def decide(
        self,
        request: ProviderInvocationRecoveryPolicyRequest,
    ) -> ProviderInvocationRecoveryPolicyDecision:
        if ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT in request.allowed_actions:
            return ProviderInvocationRecoveryPolicyDecision(
                selected_action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
            )
        return ProviderInvocationRecoveryPolicyDecision(
            selected_action=request.allowed_actions[-1],
        )

    @property
    def policy_id(self) -> str:
        return "test_select_repeat"


@dataclass
class _RecordingRepeatPort:
    calls: int = 0
    last_idempotency_key: str | None = None

    def execute_idempotent_repeat(
        self,
        *,
        original_invocation: ProviderInvocation,
        original_outcome: ProviderInvocationOutcome,
        effect_contract: object,
    ) -> ProviderInvocationRecoveryRepeatResult:
        _ = original_outcome, effect_contract
        self.calls += 1
        self.last_idempotency_key = original_invocation.idempotency_key
        return ProviderInvocationRecoveryRepeatResult(
            repeat_invocation_id=f"{original_invocation.invocation_id}-repeat",
            idempotency_key=original_invocation.idempotency_key or "",
            provider_mutation_count=1,
        )


@dataclass
class _RecordingHitlPort:
    calls: int = 0
    continuation_id: str = "gcr-test"

    def surface_hitl(self, escalation: object) -> ProviderInvocationRecoveryHitlResult:
        from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
            ProviderInvocationRecoveryEscalationContext,
        )

        _ = escalation
        self.calls += 1
        assert isinstance(escalation, ProviderInvocationRecoveryEscalationContext)
        return ProviderInvocationRecoveryHitlResult(
            escalation=escalation,
            governed_continuation_request_id=self.continuation_id,
        )


def _invocation(
    *,
    invocation_id: str = "inv-a7",
    idempotency_key: str | None = "idem-a7",
    external_task_id: str | None = "ext-1",
) -> ProviderInvocation:
    return ProviderInvocation(
        invocation_id=invocation_id,
        provider_id="gec3_deterministic_fake",
        operation="external_work.create_work",
        task_id="task-a7",
        run_id="run-a7",
        external_task_id=external_task_id,
        idempotency_key=idempotency_key,
        request_digest="sha256:" + ("ab" * 32),
        started_at=_T0,
    )


def _outcome(
    invocation_id: str,
    status: ProviderInvocationStatus,
) -> ProviderInvocationOutcome:
    return ProviderInvocationOutcome(
        invocation_id=invocation_id,
        status=status,
        completed_at=_T0,
        response_digest="sha256:" + ("cd" * 32),
    )


def _repeat_eligibility(
    invocation: ProviderInvocation,
    outcome: ProviderInvocationOutcome,
) -> ExternalEffectRepeatEligibilityResult:
    return evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=invocation,
            outcome=outcome,
            effect_contract=_CREATE_CONTRACT,
        ),
        policy=_AllowRepeatPolicy(),
    )


def _recovery_stack() -> GovernedExternalWorkProviderRecovery:
    fake = DeterministicExternalWorkFake()
    recon = GovernedExternalWorkProviderReconciliation.build(fake)
    return GovernedExternalWorkProviderRecovery.build(recon)


def test_outcome_succeeded_no_action_zero_mutations() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.SUCCEEDED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
    )
    decision = evaluate_provider_invocation_recovery(request)
    assert decision.action is ProviderInvocationRecoveryAction.NO_ACTION
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(),
    )
    assert result.provider_mutation_count == 0


def test_reconciled_confirmed_succeeded_no_repeat() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    reconciliation = ProviderInvocationReconciliationResult(
        verdict=ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED,
        reason=ProviderInvocationReconciliationReason.PROBE_EXECUTED,
        invocation_id=inv.invocation_id,
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        reconciliation=reconciliation,
    )
    decision = evaluate_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.TERMINAL_SUCCESS
    assert decision.action is not ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT


def test_failed_eligible_policy_repeat_executes_one_mutation() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    eligibility = _repeat_eligibility(inv, outcome)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=eligibility,
    )
    decision = decide_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 1
    assert result.provider_mutation_count == 1
    assert repeat_port.last_idempotency_key == inv.idempotency_key


def test_failed_not_allowed_zero_repeat_calls() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=ExternalEffectRepeatEligibilityResult(
            verdict=ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED,
            reason=ExternalEffectRepeatEligibilityReason.DENIED_POLICY,
            invocation_id=inv.invocation_id,
        ),
    )
    decision = evaluate_provider_invocation_recovery(
        request,
        policy=default_fail_closed_provider_invocation_recovery_policy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.ESCALATE_HITL
    repeat_port = _RecordingRepeatPort()
    hitl = _RecordingHitlPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port, hitl=hitl),
    )
    assert repeat_port.calls == 0
    assert hitl.calls == 1
    assert result.provider_mutation_count == 0


def test_unknown_still_unknown_eligible_policy_repeat() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    eligibility = _repeat_eligibility(inv, outcome)
    reconciliation = ProviderInvocationReconciliationResult(
        verdict=ProviderInvocationReconciliationVerdict.STILL_UNKNOWN,
        reason=ProviderInvocationReconciliationReason.PROBE_EXECUTED,
        invocation_id=inv.invocation_id,
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        reconciliation=reconciliation,
        repeat_eligibility=eligibility,
    )
    decision = decide_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT
    repeat_port = _RecordingRepeatPort()
    execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 1


def test_unknown_still_unknown_not_allowed_hitl() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    reconciliation = ProviderInvocationReconciliationResult(
        verdict=ProviderInvocationReconciliationVerdict.STILL_UNKNOWN,
        reason=ProviderInvocationReconciliationReason.PROBE_EXECUTED,
        invocation_id=inv.invocation_id,
    )
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        reconciliation=reconciliation,
    )
    decision = evaluate_provider_invocation_recovery(
        request,
        policy=default_fail_closed_provider_invocation_recovery_policy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.ESCALATE_HITL


def test_reconcile_only_repeat_impossible() -> None:
    contract = ExternalEffectContract(
        contract_id="test.reconcile_only",
        operation_key="external_work.accept_quote",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
        reconciliation_probe_refs=("get_work",),
    )
    assert evaluate_unknown_uncertainty_posture(contract) is UnknownUncertaintyPosture.RECONCILE_ONLY
    inv = ProviderInvocation(
        invocation_id="inv-accept",
        provider_id="gec3_deterministic_fake",
        operation="external_work.accept_quote",
        task_id="task-a7",
        run_id="run-a7",
        external_task_id="ext-1",
        idempotency_key="idem-accept",
        request_digest="sha256:" + ("ab" * 32),
        started_at=_T0,
    )
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    eligibility = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=contract,
        ),
        policy=_AllowRepeatPolicy(),
    )
    assert eligibility.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=contract,
        repeat_eligibility=eligibility,
    )
    decision = evaluate_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is not ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT


def test_escalate_required_posture_hitl() -> None:
    contract = ExternalEffectContract(
        contract_id="test.escalate",
        operation_key="external_work.create_work",
        category=ExternalEffectCategory.INFRASTRUCTURE,
        safety=ExternalEffectSafetyCapabilities(
            idempotency=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            reconciliation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
            compensation=ExternalEffectCapabilitySupport.NOT_SUPPORTED,
        ),
    )
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=contract,
    )
    decision = evaluate_provider_invocation_recovery(request)
    assert decision.action is ProviderInvocationRecoveryAction.ESCALATE_HITL
    assert decision.reason is ProviderInvocationRecoveryReason.ESCALATE_POSTURE


def test_missing_idempotency_no_repeat() -> None:
    inv = _invocation(idempotency_key=None)
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    eligibility = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=_CREATE_CONTRACT,
        ),
        policy=_AllowRepeatPolicy(),
    )
    assert eligibility.verdict is ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=eligibility,
    )
    decision = evaluate_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is not ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT


def test_repeat_preserves_idempotency_key() -> None:
    inv = _invocation(idempotency_key="same-key")
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    eligibility = _repeat_eligibility(inv, outcome)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=eligibility,
    )
    decision = decide_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert result.repeat is not None
    assert result.repeat.idempotency_key == inv.idempotency_key


def test_policy_failure_no_mutation() -> None:
    class _ExplodingPolicy:
        def decide(
            self,
            request: ProviderInvocationRecoveryPolicyRequest,
        ) -> ProviderInvocationRecoveryPolicyDecision:
            _ = request
            raise RuntimeError("policy boom")

    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.UNKNOWN)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
    )
    decision = evaluate_provider_invocation_recovery(request, policy=_ExplodingPolicy())
    assert decision.action is ProviderInvocationRecoveryAction.ESCALATE_HITL
    assert decision.reason is ProviderInvocationRecoveryReason.POLICY_FAILED


def test_hitl_path_zero_provider_mutations() -> None:
    inv = _invocation()
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=None,
        dispatch_state=ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY,
        effect_contract=_CREATE_CONTRACT,
    )
    decision = evaluate_provider_invocation_recovery(request)
    hitl = _RecordingHitlPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(hitl=hitl),
    )
    assert decision.action is ProviderInvocationRecoveryAction.ESCALATE_HITL
    assert hitl.calls == 1
    assert result.provider_mutation_count == 0


def test_crash_ambiguity_no_automatic_repeat() -> None:
    inv = _invocation()
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=None,
        dispatch_state=ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY,
        effect_contract=_CREATE_CONTRACT,
    )
    decision = evaluate_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    assert decision.action is ProviderInvocationRecoveryAction.ESCALATE_HITL
    repeat_port = _RecordingRepeatPort()
    execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 0


def test_execution_failure_does_not_fallback() -> None:
    class _FailingRepeatPort:
        def execute_idempotent_repeat(self, **kwargs: object) -> ProviderInvocationRecoveryRepeatResult:
            _ = kwargs
            raise RuntimeError("repeat failed")

    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
    )
    decision = decide_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=_FailingRepeatPort()),
    )
    assert result.disposition is ProviderInvocationRecoveryExecutionDisposition.FAILED
    assert result.provider_mutation_count == 0


def test_repeat_creates_new_physical_invocation_id() -> None:
    inv = _invocation(invocation_id="inv-original")
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
    )
    decision = decide_provider_invocation_recovery(
        request,
        policy=_SelectIdempotentRepeatRecoveryPolicy(),
    )
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert result.repeat is not None
    assert result.repeat.repeat_invocation_id != inv.invocation_id


def test_host_recovery_stack_decide_only() -> None:
    stack = _recovery_stack()
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.SUCCEEDED)
    request = stack.build_recovery_request(
        invocation=inv,
        outcome=outcome,
        capabilities=_CAPABILITIES,
    )
    decision = stack.decide(request)
    assert decision.action is ProviderInvocationRecoveryAction.NO_ACTION
