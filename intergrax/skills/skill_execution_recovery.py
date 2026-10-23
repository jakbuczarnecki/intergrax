# © Artur Czarnecki. All rights reserved.

"""Canonical skill execution recovery signal (SKILL-X Batch C / GR-5 continuation)."""

from __future__ import annotations

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
)
from intergrax.skills.resolver import SkillResolutionError
from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_execution_task_id,
)
from intergrax.runtime.execution.active_execution_continuation_store import (
    peek_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.progress_gate import (
    load_pending_for_execution_progress,
)


def active_execution_requires_skill_recovery() -> bool:
    """
    True when the active canonical execution is in ``RESUMED`` continuation lifecycle.

    Matches ``canonical_execution_is_resumed`` — durable skill pinning must be loaded,
    not re-resolved from the current registry.
    """
    store = peek_active_execution_continuation_state_store()
    if store is None:
        return False
    task_id = peek_active_execution_task_id()
    bound = peek_active_execution_identity()
    execution_id = peek_active_execution_id()
    if task_id is None or bound is None or execution_id is None:
        return False
    run_id, attempt_id = bound
    identity = ExecutionContinuationIdentity(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    pending = load_pending_for_execution_progress(store=store, identity=identity)
    if pending is None:
        return False
    return pending.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED


def resolve_skill_execution_recovery_required(explicit: bool | None) -> bool:
    """
    Derive recovery requirement from canonical continuation lifecycle first.

    Explicit ``True`` may require recovery; explicit ``False`` cannot disable recovery
    already mandated by a confirmed ``RESUMED`` canonical execution.
    """
    try:
        if active_execution_requires_skill_recovery():
            return True
    except ExecutionContinuationError as exc:
        raise SkillResolutionError(
            "canonical execution continuation unavailable for skill recovery authority",
        ) from exc
    if explicit is True:
        return True
    return False


__all__ = [
    "active_execution_requires_skill_recovery",
    "resolve_skill_execution_recovery_required",
]
