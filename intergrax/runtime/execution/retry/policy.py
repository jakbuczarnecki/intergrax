# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-attempt retry eligibility policy (NPSC-5E/R1)."""

from __future__ import annotations

from intergrax.contracts.execution_retry import (
    ExecutionFailureKind,
    ExecutionRetryAction,
    ExecutionRetryEligibilityRequest,
    ExecutionRetryEligibilityResult,
)
from intergrax.contracts.execution_terminal import ExecutionTerminalOutcome
from intergrax.contracts.resilience_policy import FailureClass, FailureResponse
from intergrax.runtime.resilience.policy_resolver import resolve_failure_action


_RETRYABLE_KINDS = frozenset(
    {
        ExecutionFailureKind.RETRYABLE_TRANSIENT,
        ExecutionFailureKind.RETRYABLE_TIMEOUT,
    },
)


def evaluate_execution_retry_eligibility(
    request: ExecutionRetryEligibilityRequest,
) -> ExecutionRetryEligibilityResult:
    classification = request.classification

    if request.cancelled or classification.kind is ExecutionFailureKind.CANCELLED:
        return _result(ExecutionRetryAction.CANCEL, "cancelled")

    if request.terminal_outcome is ExecutionTerminalOutcome.COMPLETED:
        return _result(ExecutionRetryAction.FAIL, "terminal_success")
    if request.terminal_outcome is ExecutionTerminalOutcome.CANCELLED:
        return _result(ExecutionRetryAction.CANCEL, "terminal_cancelled")
    if request.terminal_outcome is ExecutionTerminalOutcome.FAILED:
        return _result(ExecutionRetryAction.FAIL, "terminal_failed")

    if classification.kind is ExecutionFailureKind.TERMINAL_SUCCESS:
        return _result(ExecutionRetryAction.FAIL, "terminal_success")
    if classification.kind is ExecutionFailureKind.TERMINAL_DENY:
        return _result(ExecutionRetryAction.FAIL, "terminal_deny")

    if classification.kind in {
        ExecutionFailureKind.GOVERNANCE_DENIED,
        ExecutionFailureKind.AUTHORITY_DENIED,
        ExecutionFailureKind.TRUST_DENIED,
        ExecutionFailureKind.CONTRACT_ERROR,
        ExecutionFailureKind.NON_RETRYABLE_PERMANENT,
    }:
        return _result(ExecutionRetryAction.FAIL, classification.kind.value)

    if classification.kind is ExecutionFailureKind.BUDGET_EXHAUSTED:
        return _result(ExecutionRetryAction.FAIL, "budget_exhausted")

    if classification.kind is ExecutionFailureKind.DEADLINE_EXCEEDED:
        return _result(ExecutionRetryAction.FAIL, "deadline_exceeded")

    if classification.has_unknown_side_effect and not request.side_effect_idempotency_guaranteed:
        return _result(ExecutionRetryAction.FAIL, "unknown_side_effect")

    if classification.kind is ExecutionFailureKind.UNKNOWN:
        return _result(ExecutionRetryAction.FAIL, "unknown_fail_closed")

    if classification.kind not in _RETRYABLE_KINDS:
        return _result(ExecutionRetryAction.FAIL, classification.kind.value)

    if request.attempt_number >= request.max_attempts:
        return _result(ExecutionRetryAction.FAIL, "max_attempts_exhausted")

    if (
        request.global_deadline_monotonic is not None
        and request.now_monotonic is not None
        and request.now_monotonic + request.proposed_backoff_seconds >= request.global_deadline_monotonic
    ):
        return _result(ExecutionRetryAction.FAIL, "global_deadline_exceeded")

    return ExecutionRetryEligibilityResult(
        action=ExecutionRetryAction.RETRY,
        reason=classification.reason or classification.kind.value,
        backoff_delay_seconds=request.proposed_backoff_seconds,
    )


def project_resilience_failure_kind(
    failure_class: FailureClass,
    *,
    policy_attempt: int = 0,
) -> ExecutionFailureKind:
    resolution = resolve_failure_action(failure_class, attempt=policy_attempt)
    if resolution.response is FailureResponse.ESCALATE:
        return ExecutionFailureKind.BUDGET_EXHAUSTED
    if failure_class is FailureClass.POLICY_ERROR:
        return ExecutionFailureKind.GOVERNANCE_DENIED
    if resolution.response is FailureResponse.REQUEST_HUMAN:
        return ExecutionFailureKind.GOVERNANCE_DENIED
    if resolution.response in {
        FailureResponse.RETRY,
        FailureResponse.RETRY_RUN,
        FailureResponse.RETRY_GRAPH,
        FailureResponse.RECOVERY_REBOOT,
    }:
        if failure_class is FailureClass.DEPENDENCY_ERROR:
            return ExecutionFailureKind.RETRYABLE_TRANSIENT
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if resolution.response is FailureResponse.FAIL:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    return ExecutionFailureKind.UNKNOWN


def _result(action: ExecutionRetryAction, reason: str) -> ExecutionRetryEligibilityResult:
    return ExecutionRetryEligibilityResult(action=action, reason=reason)
