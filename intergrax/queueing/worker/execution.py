# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from intergrax.contracts.idempotency_store import IdempotencyStore
from intergrax.queueing.worker.registry import TaskExecutionRegistry
from intergrax.tools.execution_models import ToolExecutionResult


class IdempotencyLockConflictError(RuntimeError):
    """
    Raised when idempotency ledger entry is currently STARTED by another execution.

    This is considered a retryable infrastructure-level exception.
    """
    pass


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
    lease_seconds: Optional[int] = None,
    completed_ttl_seconds: Optional[int] = None,
) -> ToolExecutionResult[BaseModel]:
    """
    Pure execution core for logical task dispatch.

    Ledger-based idempotency:
    - NONE -> STARTED is atomic (record_started)
    - COMPLETED is stored with ToolExecutionResult payload
    - COMPLETED result is replayed deterministically

    Does not depend on Celery.
    """

    handler = registry.get_handler(logical_task_name)

    # No idempotency configured or no idempotency_key -> execute directly.
    if idempotency_store is None or idempotency_key is None:
        return handler(
            tenant_id=tenant_id,
            run_id=run_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    ledger_key = f"{logical_task_name}:{idempotency_key}"

    # Replay if completed
    completed = idempotency_store.get_completed_result(
        tenant_id=tenant_id,
        key=ledger_key,
    )
    if completed is not None:
        return completed

    # Atomically mark STARTED (must reject concurrent NONE->STARTED)
    try:
        idempotency_store.record_started(
            tenant_id=tenant_id,
            key=ledger_key,
            lease_seconds=lease_seconds,
        )
    except RuntimeError as exc:
        # Another worker likely already recorded STARTED/COMPLETED.
        # If it completed between our checks, replay now.
        completed_after = idempotency_store.get_completed_result(
            tenant_id=tenant_id,
            key=ledger_key,
        )
        if completed_after is not None:
            return completed_after

        raise IdempotencyLockConflictError(
            f"Invocation is already in progress for key '{ledger_key}'."
        ) from exc

    # Execute handler and persist COMPLETED
    result: ToolExecutionResult[BaseModel] = handler(
        tenant_id=tenant_id,
        run_id=run_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )

    idempotency_store.record_completed(
        tenant_id=tenant_id,
        key=ledger_key,
        result=result,
        completed_ttl_seconds=completed_ttl_seconds,
    )

    return result