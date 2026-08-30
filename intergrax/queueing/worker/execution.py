# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    IdempotencyStore,
    InvocationUncertaintyError,
)
from intergrax.queueing.worker.registry import (
    BackgroundTaskHandler,
    TaskExecutionRegistry,
)
from intergrax.runtime.background_execution.bootstrap import BackgroundExecutionIdentity
from intergrax.runtime.background_execution.identity_admission import (
    assert_handler_run_id_matches_identity,
)
from intergrax.tools.execution_models import ToolExecutionResult

_DEFAULT_LEASE_SECONDS = 300


class IdempotencyLockConflictError(RuntimeError):
    """
    Raised when idempotency ledger entry is currently STARTED by another execution.

    This is considered a retryable infrastructure-level exception.
    """
    pass


def _invoke_logical_task_handler(
    handler: BackgroundTaskHandler,
    *,
    tenant_id: str,
    run_id: str,
    payload: bytes,
    idempotency_key: Optional[str],
    execution_identity: BackgroundExecutionIdentity,
) -> ToolExecutionResult[BaseModel]:
    return handler(
        tenant_id=tenant_id,
        run_id=run_id,
        payload=payload,
        idempotency_key=idempotency_key,
        execution_identity=execution_identity,
    )


class RetryableHandlerError(RuntimeError):
    """
    Raised by logical task handler to indicate transient failure.

    This signals that the failure is retryable at the execution plane level.
    """
    pass


def execute_logical_task(
    *,
    registry: TaskExecutionRegistry,
    logical_task_name: str,
    tenant_id: str,
    run_id: str,
    payload: bytes,
    idempotency_key: Optional[str],
    idempotency_store: Optional[IdempotencyStore],
    execution_identity: BackgroundExecutionIdentity,
    lease_seconds: Optional[int] = None,
    completed_ttl_seconds: Optional[int] = None,
) -> ToolExecutionResult[BaseModel]:
    """
    Pure execution core for logical task dispatch.

    Ledger-based idempotency uses typed claim/owner/lease/fence semantics:
    - ``claim`` atomically acquires ownership or classifies existing state
    - COMPLETED results replay deterministically
    - UNCERTAIN outcomes require reconciliation (not safe blind retry)

    Does not depend on Celery.
    """

    handler = registry.get_handler(logical_task_name)
    assert_handler_run_id_matches_identity(
        handler_run_id=run_id,
        execution_identity=execution_identity,
    )
    resolved_tenant_id = execution_identity.tenant_id
    resolved_run_id = str(execution_identity.run_id)

    # No idempotency configured or no idempotency_key -> execute directly.
    if idempotency_store is None or idempotency_key is None:
        return _invoke_logical_task_handler(
            handler,
            tenant_id=resolved_tenant_id,
            run_id=resolved_run_id,
            payload=payload,
            idempotency_key=idempotency_key,
            execution_identity=execution_identity,
        )

    ledger_key = f"{logical_task_name}:{idempotency_key}"
    owner_id = f"worker-attempt-{uuid4().hex}"
    lease = lease_seconds if lease_seconds is not None else _DEFAULT_LEASE_SECONDS

    claim_result = idempotency_store.claim(
        tenant_id=resolved_tenant_id,
        key=ledger_key,
        owner_id=owner_id,
        lease_seconds=lease,
    )

    if claim_result.outcome == ClaimOutcome.REPLAY_COMPLETED:
        cached = claim_result.completed_result
        if cached is None:
            cached = idempotency_store.get_completed_result(
                tenant_id=resolved_tenant_id,
                key=ledger_key,
            )
        if cached is None:
            raise RuntimeError(
                "Ledger inconsistency: COMPLETED without stored result.",
            )
        return cached

    if claim_result.outcome == ClaimOutcome.BLOCKED_ACTIVE:
        raise IdempotencyLockConflictError(
            f"Invocation is already in progress for key '{ledger_key}'.",
        )

    if claim_result.outcome == ClaimOutcome.UNCERTAIN:
        raise InvocationUncertaintyError(
            f"Invocation outcome uncertain for key '{ledger_key}'. "
            "Reconciliation required before retry.",
        )

    claim = claim_result.claim
    if claim is None:
        raise RuntimeError("Ledger inconsistency: ACQUIRED without claim.")

    result: ToolExecutionResult[BaseModel] = _invoke_logical_task_handler(
        handler,
        tenant_id=resolved_tenant_id,
        run_id=resolved_run_id,
        payload=payload,
        idempotency_key=idempotency_key,
        execution_identity=execution_identity,
    )

    idempotency_store.complete_with_claim(
        tenant_id=resolved_tenant_id,
        key=ledger_key,
        claim=claim,
        result=result,
        completed_ttl_seconds=completed_ttl_seconds,
    )

    return result
