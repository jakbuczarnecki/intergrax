# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Contained provider execution after admission (R1 + diagnostic spine R2)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import TypeVar
from uuid import uuid4

from intergrax.contracts.execution_identity import RunId
from intergrax.contracts.external_operations.attempt import (
    ExternalOperationAttempt,
    ExternalOperationAttemptLifecycle,
    mint_external_operation_attempt_id,
)
from intergrax.contracts.external_operations.evidence import (
    ExecutionFailureEvidence,
    ExternalOperationEvidence,
    ExternalOperationEvidenceKind,
    ExternalOperationFailed,
    ProviderFailureEvidence,
)
from intergrax.contracts.external_operations.execution_context import (
    ExternalOperationExecutionContext,
)
from intergrax.contracts.external_operations.provider import ExternalOperationProvider
from intergrax.contracts.external_operations.safety import (
    ExternalOperationExecutionForbiddenError,
)
from intergrax.runtime.external_operations.admission.execution_gate import (
    ExternalOperationExecutionGate,
)
from intergrax.runtime.external_operations.diagnostic.failure_classification import (
    classify_provider_exception,
    failure_kind_retryable_default,
)
from intergrax.runtime.external_operations.diagnostic.runtime_event_recorder import (
    ExternalOperationFailureRecordResult,
    RuntimeEventExternalOperationFailureRecorder,
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
    execution_context: ExternalOperationExecutionContext | None = None,
    run_id: RunId | None = None,
    failure_recorder: RuntimeEventExternalOperationFailureRecorder | None = None,
) -> tuple[T, ExternalOperationAttempt]:
    if attempt.lifecycle is not ExternalOperationAttemptLifecycle.ADMITTED:
        raise ExternalOperationExecutionForbiddenError("attempt must be ADMITTED")
    if attempt.provider_id is not None and attempt.provider_id != provider.provider_id:
        raise ExternalOperationExecutionForbiddenError("provider_id mismatch")
    bound = attempt.bind_provider(provider.provider_id)
    if execution_context is not None:
        bound = bound.bind_platform_execution(
            execution_id=execution_context.execution_id,
            attempt_id=execution_context.attempt_id,
        )
    executing = gate.begin_execution(bound)
    _ = provider.execute_admitted  # SPI binding — concrete providers implement I/O
    try:
        value = fn()
    except Exception as exc:
        failure_kind = classify_provider_exception(exc)
        failure = ExternalOperationFailed(
            attempt_id=executing.operation_attempt_id,
            provider_failure=ProviderFailureEvidence(
                provider_id=provider.provider_id,
                failure_code=failure_kind.value,
                safe_message="external provider failure",
            ),
            execution_failure=ExecutionFailureEvidence(
                failure_code=failure_kind.value,
                safe_summary="external provider returned error",
            ),
        )
        evidence = ExternalOperationEvidence(
            evidence_id=f"ext_op_ev_{uuid4().hex}",
            attempt_id=executing.operation_attempt_id,
            intent_id=executing.intent.intent_id,
            tenant_id=executing.tenant_id or executing.intent.tenant_id,
            kind=ExternalOperationEvidenceKind.PROVIDER_FAILURE,
            safe_summary="external provider returned error",
            recorded_at=datetime.now(timezone.utc),
        )
        runtime_event_refs: tuple[str, ...] = ()
        if failure_recorder is not None and execution_context is not None and run_id is not None:
            record = failure_recorder.record_failure(
                tenant_id=execution_context.tenant_id,
                task_id=execution_context.task_id,
                run_id=run_id,
                attempt_id=execution_context.attempt_id,
                execution_id=execution_context.execution_id,
                operation_attempt_id=executing.operation_attempt_id,
                provider_id=provider.provider_id,
                operation_type=executing.intent.operation_type.value,
                failure_kind=failure_kind,
                retryable=failure_kind_retryable_default(failure_kind),
                evidence_refs=(evidence.evidence_id,),
            )
            if record.event_id is not None:
                runtime_event_refs = (str(record.event_id),)
        gate.complete_failure(executing)
        raise ContainedProviderExecutionError(failure=failure, evidence=evidence) from exc
    terminal = gate.complete_success(executing)
    return value, terminal


def retry_contained_provider_attempt(
    *,
    prior: ExternalOperationAttempt,
    execution_context: ExternalOperationExecutionContext,
) -> ExternalOperationAttempt:
    """Retry isolation — new operation attempt, same runtime execution identity."""
    if prior.lifecycle not in {
        ExternalOperationAttemptLifecycle.FAILED,
        ExternalOperationAttemptLifecycle.CANCELLED,
    }:
        raise ExternalOperationExecutionForbiddenError(
            "retry requires a terminal prior attempt",
        )
    return ExternalOperationAttempt(
        operation_attempt_id=mint_external_operation_attempt_id(),
        intent=prior.intent,
        tenant_id=prior.tenant_id or prior.intent.tenant_id,
        task_id=prior.task_id or prior.intent.task_id,
        attempt_id=execution_context.attempt_id,
        execution_id=execution_context.execution_id,
        provider_id=prior.provider_id,
        lifecycle=ExternalOperationAttemptLifecycle.ADMITTED,
        admitted_at=datetime.now(timezone.utc),
    )
