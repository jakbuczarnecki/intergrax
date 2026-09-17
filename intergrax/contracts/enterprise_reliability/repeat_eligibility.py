# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""External-effect repeat (retry) eligibility — decision only (GR-7-A5)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    UnknownUncertaintyPosture,
    contract_declares_idempotency,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)

_MAX_REASON = 512


class ExternalEffectRepeatEligibilityVerdict(StrEnum):
    """Whether an idempotent external-effect repeat may be considered safe."""

    ELIGIBLE = "eligible"
    NOT_ALLOWED = "not_allowed"
    NOT_APPLICABLE = "not_applicable"


class ExternalEffectRepeatEligibilityReason(StrEnum):
    """Closed, auditable eligibility rationale — not a loose authority string."""

    ALLOWED_IDEMPOTENT_REPEAT = "allowed_idempotent_repeat"
    DENIED_ALREADY_SUCCEEDED = "denied_already_succeeded"
    DENIED_IDEMPOTENCY_NOT_SUPPORTED = "denied_idempotency_not_supported"
    DENIED_IDEMPOTENCY_KEY_MISSING = "denied_idempotency_key_missing"
    DENIED_POLICY = "denied_policy"
    DENIED_RECONCILIATION_REQUIRED = "denied_reconciliation_required"
    DENIED_ESCALATE_REQUIRED = "denied_escalate_required"
    DENIED_INVALID_STATE = "denied_invalid_state"
    DENIED_OUTCOME_MISSING = "denied_outcome_missing"
    DENIED_NO_PROVIDER_ATTEMPT = "denied_no_provider_attempt"
    DENIED_OUTCOME_STATUS = "denied_outcome_status"


@dataclass(frozen=True, slots=True)
class ExternalEffectRepeatPolicyRequest:
    """Policy inputs — durable invocation truth and effective contract only."""

    invocation: ProviderInvocation
    outcome_status: ProviderInvocationStatus
    effect_contract: ExternalEffectContract
    unknown_posture: UnknownUncertaintyPosture


@dataclass(frozen=True, slots=True)
class ExternalEffectRepeatPolicyDecision:
    """Typed allow/deny from an injected repeat policy."""

    allow_repeat: bool


class ExternalEffectRepeatPolicy(Protocol):
    """Pluggable runtime policy governing idempotent external-effect repeat."""

    def decide(
        self,
        request: ExternalEffectRepeatPolicyRequest,
    ) -> ExternalEffectRepeatPolicyDecision:
        """Return whether repeat is permitted for this invocation under policy."""


@runtime_checkable
class ExternalEffectRepeatPolicyIdentifiable(Protocol):
    """Optional stable policy identity for audit trails."""

    @property
    def policy_id(self) -> str:
        """Immutable policy identifier."""


class ExternalEffectRepeatEligibilityRequest(BaseModel):
    """Inputs for repeat eligibility — no provider I/O."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation: ProviderInvocation | None = None
    outcome: ProviderInvocationOutcome | None = None
    effect_contract: ExternalEffectContract


class ExternalEffectRepeatEligibilityResult(BaseModel):
    """Typed repeat eligibility outcome bound to invocation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: ExternalEffectRepeatEligibilityVerdict
    reason: ExternalEffectRepeatEligibilityReason
    invocation_id: str | None = None
    idempotency_key: str | None = None
    unknown_posture: UnknownUncertaintyPosture | None = None
    policy_id: str | None = None
    detail: str = Field(default="", max_length=_MAX_REASON)


def external_effect_repeat_policy_id(
    policy: ExternalEffectRepeatPolicy | None,
) -> str | None:
    if policy is None:
        return None
    if isinstance(policy, ExternalEffectRepeatPolicyIdentifiable):
        return policy.policy_id
    return type(policy).__name__


def evaluate_external_effect_repeat_eligibility(
    request: ExternalEffectRepeatEligibilityRequest,
    *,
    policy: ExternalEffectRepeatPolicy | None = None,
) -> ExternalEffectRepeatEligibilityResult:
    """
    Decide whether repeating the external effect is safe and allowed.

    Fail-closed by default. Does not invoke providers or persist state.
    """
    invocation = request.invocation
    if invocation is None:
        return _not_applicable(
            ExternalEffectRepeatEligibilityReason.DENIED_NO_PROVIDER_ATTEMPT,
            detail="no durable provider invocation",
        )

    outcome = request.outcome
    if outcome is None:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_OUTCOME_MISSING,
            detail="provider outcome not durably recorded",
        )

    if outcome.invocation_id != invocation.invocation_id:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_INVALID_STATE,
            detail="outcome invocation_id mismatch",
        )

    contract = request.effect_contract
    if contract.operation_key != invocation.operation:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_INVALID_STATE,
            detail="effect contract operation_key mismatch",
        )

    if outcome.status is ProviderInvocationStatus.SUCCEEDED:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_ALREADY_SUCCEEDED,
            detail="effect already confirmed",
        )

    if outcome.status not in {
        ProviderInvocationStatus.FAILED,
        ProviderInvocationStatus.UNKNOWN,
    }:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_OUTCOME_STATUS,
            detail=f"unsupported outcome status {outcome.status.value}",
        )

    posture = evaluate_unknown_uncertainty_posture(contract)

    if not contract_declares_idempotency(contract):
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_NOT_SUPPORTED,
            unknown_posture=posture,
            detail="effective contract idempotency not supported",
        )

    if outcome.status is ProviderInvocationStatus.UNKNOWN:
        if posture is UnknownUncertaintyPosture.RECONCILE_ONLY:
            return _deny(
                invocation,
                ExternalEffectRepeatEligibilityReason.DENIED_RECONCILIATION_REQUIRED,
                unknown_posture=posture,
                detail="unknown requires reconciliation before repeat",
            )
        if posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED:
            return _deny(
                invocation,
                ExternalEffectRepeatEligibilityReason.DENIED_ESCALATE_REQUIRED,
                unknown_posture=posture,
                detail="unknown requires escalation",
            )

    idempotency_key = invocation.idempotency_key
    if idempotency_key is None or not idempotency_key.strip():
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_IDEMPOTENCY_KEY_MISSING,
            unknown_posture=posture,
        )

    policy_id = external_effect_repeat_policy_id(policy)
    if policy is None:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_POLICY,
            unknown_posture=posture,
            policy_id=policy_id,
            detail="no repeat policy configured",
        )

    policy_request = ExternalEffectRepeatPolicyRequest(
        invocation=invocation,
        outcome_status=outcome.status,
        effect_contract=contract,
        unknown_posture=posture,
    )
    try:
        policy_decision = policy.decide(policy_request)
    except Exception:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_POLICY,
            unknown_posture=posture,
            policy_id=policy_id,
            detail="policy evaluation failed",
        )

    if not policy_decision.allow_repeat:
        return _deny(
            invocation,
            ExternalEffectRepeatEligibilityReason.DENIED_POLICY,
            unknown_posture=posture,
            policy_id=policy_id,
        )

    return ExternalEffectRepeatEligibilityResult(
        verdict=ExternalEffectRepeatEligibilityVerdict.ELIGIBLE,
        reason=ExternalEffectRepeatEligibilityReason.ALLOWED_IDEMPOTENT_REPEAT,
        invocation_id=invocation.invocation_id,
        idempotency_key=idempotency_key,
        unknown_posture=posture,
        policy_id=policy_id,
        detail="idempotent repeat eligible under contract and policy",
    )


def _not_applicable(
    reason: ExternalEffectRepeatEligibilityReason,
    *,
    detail: str = "",
) -> ExternalEffectRepeatEligibilityResult:
    return ExternalEffectRepeatEligibilityResult(
        verdict=ExternalEffectRepeatEligibilityVerdict.NOT_APPLICABLE,
        reason=reason,
        detail=detail[:_MAX_REASON],
    )


def _deny(
    invocation: ProviderInvocation,
    reason: ExternalEffectRepeatEligibilityReason,
    *,
    unknown_posture: UnknownUncertaintyPosture | None = None,
    policy_id: str | None = None,
    detail: str = "",
) -> ExternalEffectRepeatEligibilityResult:
    return ExternalEffectRepeatEligibilityResult(
        verdict=ExternalEffectRepeatEligibilityVerdict.NOT_ALLOWED,
        reason=reason,
        invocation_id=invocation.invocation_id,
        unknown_posture=unknown_posture,
        policy_id=policy_id,
        detail=detail[:_MAX_REASON],
    )


__all__ = [
    "ExternalEffectRepeatEligibilityReason",
    "ExternalEffectRepeatEligibilityRequest",
    "ExternalEffectRepeatEligibilityResult",
    "ExternalEffectRepeatEligibilityVerdict",
    "ExternalEffectRepeatPolicy",
    "ExternalEffectRepeatPolicyDecision",
    "ExternalEffectRepeatPolicyIdentifiable",
    "ExternalEffectRepeatPolicyRequest",
    "evaluate_external_effect_repeat_eligibility",
    "external_effect_repeat_policy_id",
]
