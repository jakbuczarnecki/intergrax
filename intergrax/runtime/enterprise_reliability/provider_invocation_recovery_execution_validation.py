# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recovery decision ↔ request execution integrity gates (GR-7-A7-R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.enterprise_reliability.effect_contract import (
    UnknownUncertaintyPosture,
    evaluate_unknown_uncertainty_posture,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationVerdict,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryRequest,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityVerdict,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus


class ProviderInvocationRecoveryExecutionBlockReason(StrEnum):
    """Typed execution denial — not a substitute for recovery decision authority."""

    DECISION_ACTION_MISMATCH = "decision_action_mismatch"
    INVOCATION_MISSING = "invocation_missing"
    DECISION_INVOCATION_MISMATCH = "decision_invocation_mismatch"
    DECISION_IDEMPOTENCY_MISMATCH = "decision_idempotency_mismatch"
    IDEMPOTENCY_KEY_MISSING = "idempotency_key_missing"
    CRASH_AMBIGUITY = "crash_ambiguity"
    OUTCOME_MISSING = "outcome_missing"
    OUTCOME_INVOCATION_MISMATCH = "outcome_invocation_mismatch"
    OUTCOME_ALREADY_SUCCEEDED = "outcome_already_succeeded"
    RECONCILED_SUCCEEDED = "reconciled_succeeded"
    RECONCILIATION_INVOCATION_MISMATCH = "reconciliation_invocation_mismatch"
    REPEAT_ELIGIBILITY_MISSING = "repeat_eligibility_missing"
    REPEAT_NOT_ELIGIBLE = "repeat_not_eligible"
    ELIGIBILITY_INVOCATION_MISMATCH = "eligibility_invocation_mismatch"
    ELIGIBILITY_IDEMPOTENCY_MISMATCH = "eligibility_idempotency_mismatch"
    POSTURE_RECONCILE_ONLY = "posture_reconcile_only"
    POSTURE_ESCALATE_REQUIRED = "posture_escalate_required"
    CONTRACT_OPERATION_MISMATCH = "contract_operation_mismatch"
    REPEAT_RESULT_INVOCATION_UNCHANGED = "repeat_result_invocation_unchanged"
    REPEAT_RESULT_IDEMPOTENCY_MISMATCH = "repeat_result_idempotency_mismatch"
    REPEAT_RESULT_MUTATION_COUNT_INVALID = "repeat_result_mutation_count_invalid"


@dataclass(frozen=True, slots=True)
class ProviderInvocationRecoveryExecutionValidation:
    allowed: bool
    block_reason: ProviderInvocationRecoveryExecutionBlockReason | None = None


def _normalized_key(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def validate_provider_invocation_recovery_execution(
    request: ProviderInvocationRecoveryRequest,
    decision: ProviderInvocationRecoveryDecision,
    *,
    expected_action: ProviderInvocationRecoveryAction,
) -> ProviderInvocationRecoveryExecutionValidation:
    """Confirm decision is bound to current request before consequential execution."""
    if decision.action is not expected_action:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.DECISION_ACTION_MISMATCH,
        )

    invocation = request.invocation
    if invocation is None:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.INVOCATION_MISSING,
        )

    decision_invocation_id = _normalized_key(decision.invocation_id)
    if decision_invocation_id != invocation.invocation_id:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.DECISION_INVOCATION_MISMATCH,
        )

    if expected_action is not ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT:
        return ProviderInvocationRecoveryExecutionValidation(allowed=True)

    invocation_key = _normalized_key(invocation.idempotency_key)
    decision_key = _normalized_key(decision.idempotency_key)
    if invocation_key is None:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.IDEMPOTENCY_KEY_MISSING,
        )
    if decision_key != invocation_key:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.DECISION_IDEMPOTENCY_MISMATCH,
        )

    if (
        request.dispatch_state
        is ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY
    ):
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.CRASH_AMBIGUITY,
        )

    outcome = request.outcome
    if outcome is None:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.OUTCOME_MISSING,
        )
    if outcome.invocation_id != invocation.invocation_id:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.OUTCOME_INVOCATION_MISMATCH,
        )
    if outcome.status is ProviderInvocationStatus.SUCCEEDED:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.OUTCOME_ALREADY_SUCCEEDED,
        )

    reconciliation = request.reconciliation
    if reconciliation is not None:
        if reconciliation.invocation_id != invocation.invocation_id:
            return ProviderInvocationRecoveryExecutionValidation(
                allowed=False,
                block_reason=(
                    ProviderInvocationRecoveryExecutionBlockReason.RECONCILIATION_INVOCATION_MISMATCH
                ),
            )
        if (
            reconciliation.verdict
            is ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED
        ):
            return ProviderInvocationRecoveryExecutionValidation(
                allowed=False,
                block_reason=ProviderInvocationRecoveryExecutionBlockReason.RECONCILED_SUCCEEDED,
            )

    contract = request.effect_contract
    if contract.operation_key != invocation.operation:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.CONTRACT_OPERATION_MISMATCH,
        )

    posture = evaluate_unknown_uncertainty_posture(contract)
    if posture is UnknownUncertaintyPosture.RECONCILE_ONLY:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.POSTURE_RECONCILE_ONLY,
        )
    if posture is UnknownUncertaintyPosture.ESCALATE_REQUIRED:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.POSTURE_ESCALATE_REQUIRED,
        )

    repeat_eligibility = request.repeat_eligibility
    if repeat_eligibility is None:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.REPEAT_ELIGIBILITY_MISSING,
        )
    if repeat_eligibility.verdict is not ExternalEffectRepeatEligibilityVerdict.ELIGIBLE:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.REPEAT_NOT_ELIGIBLE,
        )
    eligibility_invocation_id = _normalized_key(repeat_eligibility.invocation_id)
    if eligibility_invocation_id != invocation.invocation_id:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.ELIGIBILITY_INVOCATION_MISMATCH,
        )
    eligibility_key = _normalized_key(repeat_eligibility.idempotency_key)
    if eligibility_key != invocation_key:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.ELIGIBILITY_IDEMPOTENCY_MISMATCH,
        )

    return ProviderInvocationRecoveryExecutionValidation(allowed=True)


def validate_idempotent_repeat_port_result(
    *,
    original_invocation_id: str,
    original_idempotency_key: str,
    repeat_invocation_id: str,
    repeat_idempotency_key: str,
    provider_mutation_count: int,
) -> ProviderInvocationRecoveryExecutionValidation:
    """Post-conditions for a successful repeat port invocation."""
    if repeat_invocation_id == original_invocation_id or not repeat_invocation_id.strip():
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.REPEAT_RESULT_INVOCATION_UNCHANGED,
        )
    if repeat_idempotency_key != original_idempotency_key:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.REPEAT_RESULT_IDEMPOTENCY_MISMATCH,
        )
    if provider_mutation_count != 1:
        return ProviderInvocationRecoveryExecutionValidation(
            allowed=False,
            block_reason=ProviderInvocationRecoveryExecutionBlockReason.REPEAT_RESULT_MUTATION_COUNT_INVALID,
        )
    return ProviderInvocationRecoveryExecutionValidation(allowed=True)


__all__ = [
    "ProviderInvocationRecoveryExecutionBlockReason",
    "ProviderInvocationRecoveryExecutionValidation",
    "validate_idempotent_repeat_port_result",
    "validate_provider_invocation_recovery_execution",
]
