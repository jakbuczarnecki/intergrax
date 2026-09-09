# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-attempt failure classification adapter (NPSC-5E/R1)."""

from __future__ import annotations

from intergrax.contracts.execution_retry import (
    ExecutionFailureClassification,
    ExecutionFailureKind,
)
from intergrax.contracts.resilience_policy import FailureClass, FailureResponse


def classify_from_failure_class(failure_class: FailureClass) -> ExecutionFailureKind:
    if failure_class is FailureClass.USER_ERROR:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    if failure_class is FailureClass.POLICY_ERROR:
        return ExecutionFailureKind.GOVERNANCE_DENIED
    if failure_class is FailureClass.DEPENDENCY_ERROR:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if failure_class is FailureClass.QUALITY_ERROR:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if failure_class is FailureClass.RUNTIME_ERROR:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    return ExecutionFailureKind.UNKNOWN


def classify_from_failure_response(response: FailureResponse) -> ExecutionFailureKind:
    if response is FailureResponse.RETRY:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if response in {
        FailureResponse.RETRY_RUN,
        FailureResponse.RETRY_GRAPH,
        FailureResponse.RECOVERY_REBOOT,
    }:
        return ExecutionFailureKind.RETRYABLE_TRANSIENT
    if response is FailureResponse.REQUEST_HUMAN:
        return ExecutionFailureKind.GOVERNANCE_DENIED
    if response is FailureResponse.RETRY_ALTERNATE:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    if response is FailureResponse.FAIL:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    if response is FailureResponse.CIRCUIT_BREAK:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    if response in {FailureResponse.DEGRADE, FailureResponse.PARTIAL}:
        return ExecutionFailureKind.NON_RETRYABLE_PERMANENT
    if response is FailureResponse.ESCALATE:
        return ExecutionFailureKind.BUDGET_EXHAUSTED
    return ExecutionFailureKind.UNKNOWN


def classify_execution_failure(
    *,
    kind: ExecutionFailureKind,
    reason: str = "",
    failure_class: FailureClass | None = None,
    has_unknown_side_effect: bool = False,
) -> ExecutionFailureClassification:
    return ExecutionFailureClassification(
        kind=kind,
        reason=reason,
        failure_class=failure_class,
        has_unknown_side_effect=has_unknown_side_effect,
    )
