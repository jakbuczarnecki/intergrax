# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contained provider execution after admission (R1)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

from intergrax.contracts.external_operations.attempt import (
    ExternalOperationAttempt,
    ExternalOperationAttemptLifecycle,
)
from intergrax.contracts.external_operations.evidence import (
    ExecutionFailureEvidence,
    ExternalOperationEvidence,
    ExternalOperationEvidenceKind,
    ExternalOperationFailed,
    ProviderFailureEvidence,
)
from intergrax.contracts.external_operations.provider import ExternalOperationProvider
from intergrax.contracts.external_operations.safety import (
    ExternalOperationExecutionForbiddenError,
)
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)

T = TypeVar("T")


class ContainedProviderExecutionError(RuntimeError):
    def __init__(
        self, *, failure: ExternalOperationFailed, evidence: ExternalOperationEvidence
    ) -> None:
        summary = (
            failure.execution_failure.safe_summary
            if failure.execution_failure
            else "provider failed"
        )
        super().__init__(summary)
        self.failure = failure
        self.evidence = evidence


def execute_contained_provider_call(
    *,
    gate: ExternalOperationExecutionGate,
    provider: ExternalOperationProvider,
    attempt: ExternalOperationAttempt,
    fn: Callable[[], T],
) -> tuple[T, ExternalOperationAttempt]:
    if attempt.lifecycle is not ExternalOperationAttemptLifecycle.ADMITTED:
        raise ExternalOperationExecutionForbiddenError("attempt must be ADMITTED")
    if attempt.provider_id is not None and attempt.provider_id != provider.provider_id:
        raise ExternalOperationExecutionForbiddenError("provider_id mismatch")
    executing = gate.begin_execution(attempt.bind_provider(provider.provider_id))
    _ = provider.execute_admitted  # SPI binding — concrete providers implement I/O
    try:
        value = fn()
    except Exception as exc:
        failure = ExternalOperationFailed(
            attempt_id=executing.attempt_id,
            provider_failure=ProviderFailureEvidence(
                provider_id=provider.provider_id,
                failure_code=type(exc).__name__,
                safe_message=str(exc)[:512],
            ),
            execution_failure=ExecutionFailureEvidence(
                failure_code="provider_invoke_failed",
                safe_summary="external provider returned error",
            ),
        )
        evidence = ExternalOperationEvidence(
            evidence_id=f"ext_op_ev_{uuid4().hex}",
            attempt_id=executing.attempt_id,
            intent_id=executing.intent.intent_id,
            tenant_id=executing.intent.tenant_id,
            kind=ExternalOperationEvidenceKind.PROVIDER_FAILURE,
            safe_summary="external provider returned error",
            recorded_at=datetime.now(timezone.utc),
        )
        gate.complete_failure(executing)
        raise ContainedProviderExecutionError(failure=failure, evidence=evidence) from exc
    terminal = gate.complete_success(executing)
    return value, terminal
