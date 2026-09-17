# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Provider invocation controlled recovery — decision only (GR-7-A7)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    UnknownUncertaintyPosture,
    contract_declares_reconciliation,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationResult,
    ProviderInvocationReconciliationVerdict,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityResult,
    ExternalEffectRepeatEligibilityVerdict,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)

_MAX_REASON = 512


class ProviderInvocationRecoveryDispatchState(StrEnum):
    """Durable dispatch truth — distinct from UNKNOWN outcome classification."""

    OUTCOME_RECORDED = "outcome_recorded"
    CRASH_AMBIGUITY = "crash_ambiguity"


class ProviderInvocationRecoveryAction(StrEnum):
    """Single selected recovery action — no provider semantics in names."""

    NO_ACTION = "no_action"
    TERMINAL_SUCCESS = "terminal_success"
    TERMINAL_FAILURE = "terminal_failure"
    RECONCILE = "reconcile"
    IDEMPOTENT_REPEAT = "idempotent_repeat"
    ESCALATE_HITL = "escalate_hitl"


class ProviderInvocationRecoveryReason(StrEnum):
    """Closed recovery rationale."""

    ALREADY_SUCCEEDED = "already_succeeded"
    RECONCILED_SUCCEEDED = "reconciled_succeeded"
    TERMINAL_FAILURE = "terminal_failure"
    CRASH_AMBIGUITY = "crash_ambiguity"
    ESCALATE_POSTURE = "escalate_posture"
    RECONCILE_ONLY = "reconcile_only"
    REPEAT_NOT_ELIGIBLE = "repeat_not_eligible"
    REPEAT_ELIGIBILITY_MISSING = "repeat_eligibility_missing"
    POLICY_SELECTED = "policy_selected"
    POLICY_DENIED = "policy_denied"
    POLICY_FAILED = "policy_failed"
    POLICY_MISSING = "policy_missing"
    DEFAULT_FAIL_CLOSED = "default_fail_closed"
    INVALID_STATE = "invalid_state"


@dataclass(frozen=True, slots=True)
class ProviderInvocationRecoveryPolicyRequest:
    """Policy inputs — durable facts and candidate actions only."""

    invocation: ProviderInvocation
    outcome: ProviderInvocationOutcome | None
    dispatch_state: ProviderInvocationRecoveryDispatchState
    effect_contract: ExternalEffectContract
    unknown_posture: UnknownUncertaintyPosture
    reconciliation: ProviderInvocationReconciliationResult | None
    repeat_eligibility: ExternalEffectRepeatEligibilityResult | None
    allowed_actions: tuple[ProviderInvocationRecoveryAction, ...]


@dataclass(frozen=True, slots=True)
class ProviderInvocationRecoveryPolicyDecision:
    """Typed action choice from an injected recovery policy."""

    selected_action: ProviderInvocationRecoveryAction


@runtime_checkable
class ProviderInvocationRecoveryPolicyIdentifiable(Protocol):
    @property
    def policy_id(self) -> str:
        """Immutable policy identifier."""


class ProviderInvocationRecoveryPolicy(Protocol):
    """Pluggable recovery action selection — must stay within allowed_actions."""

    def decide(
        self,
        request: ProviderInvocationRecoveryPolicyRequest,
    ) -> ProviderInvocationRecoveryPolicyDecision:
        """Select one allowed recovery action."""


class ProviderInvocationRecoveryRequest(BaseModel):
    """Canonical recovery decision inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation: ProviderInvocation | None = None
    outcome: ProviderInvocationOutcome | None = None
    dispatch_state: ProviderInvocationRecoveryDispatchState = (
        ProviderInvocationRecoveryDispatchState.OUTCOME_RECORDED
    )
    effect_contract: ExternalEffectContract
    reconciliation: ProviderInvocationReconciliationResult | None = None
    repeat_eligibility: ExternalEffectRepeatEligibilityResult | None = None


class ProviderInvocationRecoveryDecision(BaseModel):
    """Pure recovery authority outcome for one evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ProviderInvocationRecoveryAction
    reason: ProviderInvocationRecoveryReason
    invocation_id: str | None = None
    idempotency_key: str | None = None
    unknown_posture: UnknownUncertaintyPosture | None = None
    policy_id: str | None = None
    detail: str = Field(default="", max_length=_MAX_REASON)


class ProviderInvocationRecoveryEscalationContext(BaseModel):
    """Typed HITL payload — maps to GovernedContinuation / HumanRequest composition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation: ProviderInvocation
    outcome: ProviderInvocationOutcome | None = None
    dispatch_state: ProviderInvocationRecoveryDispatchState
    effect_contract_id: str = Field(min_length=1, max_length=256)
    recovery_reason: ProviderInvocationRecoveryReason
    reconciliation_invocation_id: str | None = None
    reconciliation_verdict: ProviderInvocationReconciliationVerdict | None = None
    repeat_eligibility_verdict: ExternalEffectRepeatEligibilityVerdict | None = None
    repeat_eligibility_reason: str | None = Field(default=None, max_length=_MAX_REASON)
    detail: str = Field(default="", max_length=_MAX_REASON)


def provider_invocation_recovery_policy_id(
    policy: ProviderInvocationRecoveryPolicy | None,
) -> str | None:
    if policy is None:
        return None
    if isinstance(policy, ProviderInvocationRecoveryPolicyIdentifiable):
        return policy.policy_id
    return type(policy).__name__


def evaluate_provider_invocation_recovery(
    request: ProviderInvocationRecoveryRequest,
    *,
    policy: ProviderInvocationRecoveryPolicy | None = None,
) -> ProviderInvocationRecoveryDecision:
    """
    Fail-closed recovery decision from durable provider truth.

    Does not invoke providers, reconcile, repeat, or escalate.
    """
    invocation = request.invocation
    if invocation is None:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.INVALID_STATE,
            detail="provider invocation required",
        )

    dispatch_state = request.dispatch_state
    if dispatch_state is ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.CRASH_AMBIGUITY,
            invocation_id=invocation.invocation_id,
            detail="intent without durable outcome",
        )

    outcome = request.outcome
    if outcome is None:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.CRASH_AMBIGUITY,
            invocation_id=invocation.invocation_id,
            detail="outcome missing under outcome_recorded dispatch state",
        )

    if outcome.invocation_id != invocation.invocation_id:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.INVALID_STATE,
            invocation_id=invocation.invocation_id,
            detail="outcome invocation_id mismatch",
        )

    contract = request.effect_contract
    if contract.operation_key != invocation.operation:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.INVALID_STATE,
            invocation_id=invocation.invocation_id,
            detail="effect contract operation_key mismatch",
        )

    posture = evaluate_unknown_uncertainty_posture(contract)
    reconciliation = request.reconciliation
    repeat_eligibility = request.repeat_eligibility

    if outcome.status is ProviderInvocationStatus.SUCCEEDED:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.NO_ACTION,
            reason=ProviderInvocationRecoveryReason.ALREADY_SUCCEEDED,
            invocation_id=invocation.invocation_id,
            idempotency_key=invocation.idempotency_key,
            unknown_posture=posture,
        )

    if (
        reconciliation is not None
        and reconciliation.verdict
        is ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED
    ):
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.TERMINAL_SUCCESS,
            reason=ProviderInvocationRecoveryReason.RECONCILED_SUCCEEDED,
            invocation_id=invocation.invocation_id,
            idempotency_key=invocation.idempotency_key,
            unknown_posture=posture,
        )

    if posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.ESCALATE_POSTURE,
            invocation_id=invocation.invocation_id,
            unknown_posture=posture,
        )

    allowed = _allowed_recovery_actions(
        invocation=invocation,
        outcome=outcome,
        contract=contract,
        posture=posture,
        reconciliation=reconciliation,
        repeat_eligibility=repeat_eligibility,
    )

    if len(allowed) == 1:
        return _decision_for_action(
            allowed[0],
            invocation=invocation,
            posture=posture,
            reason=_single_action_reason(allowed[0], outcome=outcome),
        )

    policy_id = provider_invocation_recovery_policy_id(policy)
    if policy is None:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.POLICY_MISSING,
            invocation_id=invocation.invocation_id,
            unknown_posture=posture,
            policy_id=policy_id,
            detail="recovery policy required for ambiguous recovery",
        )

    policy_request = ProviderInvocationRecoveryPolicyRequest(
        invocation=invocation,
        outcome=outcome,
        dispatch_state=dispatch_state,
        effect_contract=contract,
        unknown_posture=posture,
        reconciliation=reconciliation,
        repeat_eligibility=repeat_eligibility,
        allowed_actions=allowed,
    )
    try:
        policy_decision = policy.decide(policy_request)
    except Exception:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.POLICY_FAILED,
            invocation_id=invocation.invocation_id,
            unknown_posture=posture,
            policy_id=policy_id,
        )

    selected = policy_decision.selected_action
    if selected not in allowed:
        return ProviderInvocationRecoveryDecision(
            action=ProviderInvocationRecoveryAction.ESCALATE_HITL,
            reason=ProviderInvocationRecoveryReason.POLICY_DENIED,
            invocation_id=invocation.invocation_id,
            unknown_posture=posture,
            policy_id=policy_id,
            detail="policy selected disallowed action",
        )

    return _decision_for_action(
        selected,
        invocation=invocation,
        posture=posture,
        reason=ProviderInvocationRecoveryReason.POLICY_SELECTED,
        policy_id=policy_id,
    )


def _allowed_recovery_actions(
    *,
    invocation: ProviderInvocation,
    outcome: ProviderInvocationOutcome,
    contract: ExternalEffectContract,
    posture: UnknownUncertaintyPosture,
    reconciliation: ProviderInvocationReconciliationResult | None,
    repeat_eligibility: ExternalEffectRepeatEligibilityResult | None,
) -> tuple[ProviderInvocationRecoveryAction, ...]:
    repeat_allowed = _repeat_action_allowed(
        posture=posture,
        repeat_eligibility=repeat_eligibility,
    )
    reconcile_allowed = _reconcile_action_allowed(
        outcome=outcome,
        contract=contract,
        posture=posture,
        reconciliation=reconciliation,
    )

    if outcome.status is ProviderInvocationStatus.FAILED:
        if repeat_allowed:
            return (ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,)
        return (ProviderInvocationRecoveryAction.ESCALATE_HITL,)

    if outcome.status is not ProviderInvocationStatus.UNKNOWN:
        return (ProviderInvocationRecoveryAction.ESCALATE_HITL,)

    if (
        reconciliation is not None
        and reconciliation.verdict
        is ProviderInvocationReconciliationVerdict.CONFIRMED_FAILED
    ):
        if repeat_allowed:
            return (ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,)
        return (ProviderInvocationRecoveryAction.ESCALATE_HITL,)

    if posture is UnknownUncertaintyPosture.RECONCILE_ONLY:
        if reconcile_allowed:
            return (ProviderInvocationRecoveryAction.RECONCILE,)
        return (ProviderInvocationRecoveryAction.ESCALATE_HITL,)

    options: list[ProviderInvocationRecoveryAction] = []
    if reconcile_allowed:
        options.append(ProviderInvocationRecoveryAction.RECONCILE)
    if repeat_allowed:
        options.append(ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT)
    options.append(ProviderInvocationRecoveryAction.ESCALATE_HITL)
    return tuple(dict.fromkeys(options))


def _repeat_action_allowed(
    *,
    posture: UnknownUncertaintyPosture,
    repeat_eligibility: ExternalEffectRepeatEligibilityResult | None,
) -> bool:
    if posture is UnknownUncertaintyPosture.RECONCILE_ONLY:
        return False
    if posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED:
        return False
    if repeat_eligibility is None:
        return False
    return repeat_eligibility.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE


def _reconcile_action_allowed(
    *,
    outcome: ProviderInvocationOutcome,
    contract: ExternalEffectContract,
    posture: UnknownUncertaintyPosture,
    reconciliation: ProviderInvocationReconciliationResult | None,
) -> bool:
    if outcome.status is not ProviderInvocationStatus.UNKNOWN:
        return False
    if not contract_declares_reconciliation(contract):
        return False
    if reconciliation is not None:
        return False
    return True


def _single_action_reason(
    action: ProviderInvocationRecoveryAction,
    *,
    outcome: ProviderInvocationOutcome,
) -> ProviderInvocationRecoveryReason:
    if action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT:
        return ProviderInvocationRecoveryReason.POLICY_SELECTED
    if action is ProviderInvocationRecoveryAction.RECONCILE:
        return ProviderInvocationRecoveryReason.RECONCILE_ONLY
    if action is ProviderInvocationRecoveryAction.ESCALATE_HITL:
        if outcome.status is ProviderInvocationStatus.FAILED:
            return ProviderInvocationRecoveryReason.REPEAT_NOT_ELIGIBLE
        return ProviderInvocationRecoveryReason.DEFAULT_FAIL_CLOSED
    if action is ProviderInvocationRecoveryAction.TERMINAL_FAILURE:
        return ProviderInvocationRecoveryReason.TERMINAL_FAILURE
    return ProviderInvocationRecoveryReason.DEFAULT_FAIL_CLOSED


def _decision_for_action(
    action: ProviderInvocationRecoveryAction,
    *,
    invocation: ProviderInvocation,
    posture: UnknownUncertaintyPosture,
    reason: ProviderInvocationRecoveryReason,
    policy_id: str | None = None,
) -> ProviderInvocationRecoveryDecision:
    return ProviderInvocationRecoveryDecision(
        action=action,
        reason=reason,
        invocation_id=invocation.invocation_id,
        idempotency_key=invocation.idempotency_key,
        unknown_posture=posture,
        policy_id=policy_id,
    )


__all__ = [
    "ProviderInvocationRecoveryAction",
    "ProviderInvocationRecoveryDecision",
    "ProviderInvocationRecoveryDispatchState",
    "ProviderInvocationRecoveryEscalationContext",
    "ProviderInvocationRecoveryPolicy",
    "ProviderInvocationRecoveryPolicyDecision",
    "ProviderInvocationRecoveryPolicyIdentifiable",
    "ProviderInvocationRecoveryPolicyRequest",
    "ProviderInvocationRecoveryReason",
    "ProviderInvocationRecoveryRequest",
    "evaluate_provider_invocation_recovery",
    "provider_invocation_recovery_policy_id",
]
