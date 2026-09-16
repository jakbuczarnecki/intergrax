# © Artur Czarnecki. All rights reserved.

"""GR-5-R5 — durable process restart qualification for canonical continuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationRecoveryHandle,
    PendingExecutionContinuation,
    assert_execution_continuation_identity_match,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.execution_identity import TaskId
from intergrax.runtime.execution.continuation.execution_continuation_identity import (
    execution_identity_binding_from_continuation_identity,
)
from intergrax.runtime.execution.runtime import RootExecutionContext

if TYPE_CHECKING:
    from intergrax.runtime.execution.identity_binding import ExecutionIdentityBinding


@dataclass(frozen=True, slots=True)
class ExecutionContinuationRestartQualification:
    """Internal Execution Engine outcome after durable continuation restore."""

    identity: ExecutionContinuationIdentity
    current_episode: PendingExecutionContinuation
    execution_identity_binding: ExecutionIdentityBinding
    root_execution_context: RootExecutionContext


def recover_execution_continuation_process_restart(
    *,
    store: ExecutionContinuationStateStore,
    recovery_handle: ExecutionContinuationRecoveryHandle,
    authority: ParentExecutionAuthority,
    tenant_id: str | None = None,
    checkpoint_identity: ExecutionContinuationIdentity | None = None,
    task_id_consistency: TaskId | None = None,
    expected_identity: ExecutionContinuationIdentity | None = None,
) -> ExecutionContinuationRestartQualification:
    """
    Durable process-boundary recovery: resolve exact four-ID from persisted handle.

    Does not require caller-supplied ``ExecutionContinuationIdentity`` for discovery.
    """
    if not store.is_durable:
        raise ExecutionContinuationError(
            "continuation restart requires durable ExecutionContinuationStateStore",
            code=ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE,
        )
    continuation_id = recovery_handle.continuation_id
    if not continuation_id:
        raise ExecutionContinuationError(
            "malformed execution continuation recovery handle",
            code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
        )
    try:
        located = store.load(continuation_id)
    except ExecutionContinuationError:
        raise
    except Exception as exc:
        raise ExecutionContinuationError(
            "continuation store query failed during recovery handle resolution",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        ) from exc
    if located is None:
        raise ExecutionContinuationError(
            "recovery handle does not resolve to a continuation snapshot",
            code=ExecutionContinuationErrorCode.NOT_FOUND,
        )
    try:
        current = store.resolve_current_episode_for_identity(located.identity)
    except ExecutionContinuationError:
        raise
    except Exception as exc:
        raise ExecutionContinuationError(
            "continuation store query failed during current episode validation",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        ) from exc
    if current is None or current.continuation_id != located.continuation_id:
        raise ExecutionContinuationError(
            "recovery handle targets a non-current continuation episode",
            code=ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE,
        )
    if expected_identity is not None:
        assert_execution_continuation_identity_match(expected_identity, located.identity)
    return qualify_execution_continuation_process_restart(
        store=store,
        identity=located.identity,
        authority=authority,
        tenant_id=tenant_id,
        checkpoint_identity=checkpoint_identity,
        task_id_consistency=task_id_consistency,
    )


def qualify_execution_continuation_process_restart(
    *,
    store: ExecutionContinuationStateStore,
    identity: ExecutionContinuationIdentity,
    authority: ParentExecutionAuthority,
    tenant_id: str | None = None,
    checkpoint_identity: ExecutionContinuationIdentity | None = None,
    task_id_consistency: TaskId | None = None,
) -> ExecutionContinuationRestartQualification:
    """
    Reconstruct authoritative continuation + exact four-ID context after process restart.

    Requires a durable store provider. Does not mint AttemptId/ExecutionId or invoke
    root governance admission.
    """
    if not store.is_durable:
        raise ExecutionContinuationError(
            "continuation restart requires durable ExecutionContinuationStateStore",
            code=ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE,
        )
    try:
        current = store.resolve_current_episode_for_identity(identity)
    except ExecutionContinuationError:
        raise
    except Exception as exc:
        raise ExecutionContinuationError(
            "continuation store query failed during restart qualification",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        ) from exc
    if current is None:
        raise ExecutionContinuationError(
            "no current continuation episode for restart qualification",
            code=ExecutionContinuationErrorCode.NOT_FOUND,
        )
    assert_execution_continuation_identity_match(identity, current.identity)
    if checkpoint_identity is not None:
        assert_execution_continuation_identity_match(checkpoint_identity, current.identity)
    if task_id_consistency is not None and task_id_consistency != current.identity.task_id:
        raise ExecutionContinuationError(
            "task identity inconsistent with canonical continuation",
            code=ExecutionContinuationErrorCode.IDENTITY_MISMATCH,
        )
    binding = execution_identity_binding_from_continuation_identity(current.identity)
    root_context = RootExecutionContext(
        run_id=current.identity.run_id,
        attempt_id=current.identity.attempt_id,
        execution_id=current.identity.execution_id,
        authority=authority,
        tenant_id=tenant_id,
        task_id=current.identity.task_id,
    )
    return ExecutionContinuationRestartQualification(
        identity=current.identity,
        current_episode=current,
        execution_identity_binding=binding,
        root_execution_context=root_context,
    )


__all__ = [
    "ExecutionContinuationRestartQualification",
    "qualify_execution_continuation_process_restart",
    "recover_execution_continuation_process_restart",
]
