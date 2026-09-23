# © Artur Czarnecki. All rights reserved.

"""CLAIMED → BLOCKED authority generation transitions (UCA-6C-R6-R5.7)."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
)
from intergrax.contracts.execution.suspended_operation.authority_scope import (
    SuspendedOperationAuthorityScope,
)
from intergrax.contracts.execution.suspended_operation.claim import (
    SuspendedOperationMutationOutcome,
)
from intergrax.contracts.execution.suspended_operation.descriptor import (
    SuspendedExecutionOperationDescriptor,
    SuspendedOperationMaterializationState,
)
from intergrax.contracts.execution.suspended_operation.store import (
    SuspendedExecutionOperationStore,
)
from intergrax.contracts.governed_continuation import GovernedContinuationRequest
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
    establish_canonical_hitl_pause,
)
from intergrax.runtime.task.task import Task, TaskState


class AuthoritySequentialPauseError(RuntimeError):
    """Fail-closed sequential authority reblock."""


def ensure_prior_continuation_resumed_projection_for_replacement(
    *,
    task: Task,
    hitl_continuation: InternalOrchestrationContinuation,
    prior_continuation_id: str,
) -> None:
    """Align Task projection metadata so a new continuation episode may replace RESUMED."""
    prior_pending = hitl_continuation.port.get_pending(
        ExecutionContinuationLookup(continuation_id=prior_continuation_id),
    )
    if prior_pending.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED:
        return
    gov = task.runtime.governance
    gov.projected_continuation_id = prior_pending.continuation_id
    gov.projected_continuation_lifecycle_state = (
        ExecutionContinuationLifecycleState.RESUMED.value
    )
    gov.projected_continuation_revision = prior_pending.revision
    task.sync_metadata()


def reblock_claimed_descriptor_for_next_authority_pause(
    *,
    store: SuspendedExecutionOperationStore,
    hitl_continuation: InternalOrchestrationContinuation,
    task: Task,
    claimed: SuspendedExecutionOperationDescriptor,
    identity: ExecutionContinuationIdentity,
    governed_request: GovernedContinuationRequest,
    next_invocation_scope_id: str,
    next_authority_scope: SuspendedOperationAuthorityScope,
    pause_id: str,
    human_request_id: str,
    human_prompt: str | None = None,
    execution_interrupt: object | None = None,
) -> SuspendedExecutionOperationDescriptor:
    """CAS CLAIMED → BLOCKED for the next monotonic pause generation."""
    ownership = claimed.claim_ownership
    if ownership is None:
        raise AuthoritySequentialPauseError("reblock requires active claim ownership")
    if (
        claimed.materialization_state
        is not SuspendedOperationMaterializationState.CLAIMED
    ):
        raise AuthoritySequentialPauseError(
            f"reblock requires CLAIMED state, got {claimed.materialization_state}",
        )
    next_generation = claimed.pause_generation + 1
    correlation = governed_request.to_correlation()
    continuation_id = governed_request.continuation_request_id
    ensure_prior_continuation_resumed_projection_for_replacement(
        task=task,
        hitl_continuation=hitl_continuation,
        prior_continuation_id=claimed.continuation_id,
    )
    canonical_pending = establish_canonical_hitl_pause(
        task,
        identity=identity,
        continuation_id=continuation_id,
        reason=governed_request.reason,
        pause_id=pause_id,
        human_request_id=human_request_id,
        capability=hitl_continuation,
        governed_correlation=correlation,
        human_prompt=human_prompt,
        execution_interrupt=execution_interrupt,
    )
    if canonical_pending.lifecycle_state not in {
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
    }:
        raise AuthoritySequentialPauseError(
            "canonical pause did not reach human-waiting state after reblock",
        )
    reblocked = store.authority_reblock_from_claimed(
        suspended_operation_id=claimed.suspended_operation_id,
        expected_materialization_revision=claimed.materialization_revision,
        expected_pause_generation=claimed.pause_generation,
        expected_owner_id=ownership.owner_id,
        expected_fence=ownership.fence,
        next_pause_generation=next_generation,
        next_continuation=canonical_pending,
        next_governed_correlation=correlation,
        next_invocation_scope_id=next_invocation_scope_id,
        next_authority_scope=next_authority_scope,
    )
    if reblocked.outcome is not SuspendedOperationMutationOutcome.APPLIED:
        raise AuthoritySequentialPauseError(
            f"authority reblock failed: {reblocked.outcome.value}",
        )
    if reblocked.descriptor is None:
        raise AuthoritySequentialPauseError("reblock missing descriptor")
    task.state = TaskState.WAITING_FOR_HUMAN
    task.sync_metadata()
    return reblocked.descriptor


__all__ = [
    "AuthoritySequentialPauseError",
    "ensure_prior_continuation_resumed_projection_for_replacement",
    "reblock_claimed_descriptor_for_next_authority_pause",
]
