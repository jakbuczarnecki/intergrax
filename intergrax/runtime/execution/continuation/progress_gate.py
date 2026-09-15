# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R1 — canonical Execution progress gate bound to continuation lifecycle."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
    execution_continuation_lifecycle_blocks_execution_progress,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore

ResultT = TypeVar("ResultT")


def continuation_snapshot_blocks_execution_progress(
    pending: PendingExecutionContinuation | None,
) -> bool:
    if pending is None:
        return False
    return execution_continuation_lifecycle_blocks_execution_progress(pending.lifecycle_state)


def load_pending_for_execution_progress(
    *,
    store: ExecutionContinuationStateStore,
    identity: ExecutionContinuationIdentity,
) -> PendingExecutionContinuation | None:
    try:
        return store.resolve_identity_for_execution_progress(identity)
    except ExecutionContinuationError:
        raise
    except Exception as exc:
        raise ExecutionContinuationError(
            "continuation store query failed",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        ) from exc


def assert_canonical_execution_may_progress(
    *,
    store: ExecutionContinuationStateStore,
    identity: ExecutionContinuationIdentity,
) -> None:
    """Fail closed when continuation lifecycle blocks the active canonical Execution."""
    pending = load_pending_for_execution_progress(store=store, identity=identity)
    if pending is None:
        return
    if execution_continuation_lifecycle_blocks_execution_progress(pending.lifecycle_state):
        raise ExecutionContinuationError(
            f"execution progress blocked by continuation {pending.lifecycle_state}",
            code=ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED,
        )


async def execute_canonical_work_when_unblocked(
    *,
    store: ExecutionContinuationStateStore,
    identity: ExecutionContinuationIdentity,
    work: Callable[[], Awaitable[ResultT]],
) -> ResultT:
    """Single Execution Engine progress boundary — consult continuation then run work."""
    assert_canonical_execution_may_progress(store=store, identity=identity)
    return await work()


__all__ = [
    "assert_canonical_execution_may_progress",
    "continuation_snapshot_blocks_execution_progress",
    "execute_canonical_work_when_unblocked",
    "load_pending_for_execution_progress",
]
