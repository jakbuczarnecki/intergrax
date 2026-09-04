# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision durable recovery helpers hosted by canonical Execution (DS-REC-02/03)."""

from __future__ import annotations

from typing import Generic, TypeVar

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    decision_checkpoint_state,
    restore_decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import (
    DecisionFinalizationKey,
    DecisionFinalizeGuardState,
    decision_finalization_key,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
    initial_decision_lifecycle_state,
    transition_decision_lifecycle,
)
from intergrax.contracts.decision_revision import (
    DecisionRevisionCheckpointState,
    DecisionRevisionPolicy,
    DecisionRevisionState,
    revision_policy_from_checkpoint,
    revision_state_from_checkpoint,
    validate_resume_revision_policy,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import (
    DecisionCheckpointPersistence,
    load_decision_checkpoint,
    save_decision_checkpoint,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionFinalizationPersistence,
    load_decision_finalization_guard_state,
)

T = TypeVar("T")


class DecisionCheckpointCorruptionError(ValueError):
    """Raised when a durable checkpoint cannot be restored safely."""


def resume_decision_checkpoint_state(
    checkpoint: DecisionCheckpointState[T],
    *,
    runtime_revision_policy: DecisionRevisionPolicy | None = None,
) -> DecisionCheckpointState[T]:
    """Validate and restore one checkpoint without resetting semantic budgets."""
    restored = restore_decision_checkpoint_state(checkpoint)
    if restored.revision is not None and runtime_revision_policy is not None:
        validate_resume_revision_policy(
            checkpoint_revision=restored.revision,
            runtime_policy=runtime_revision_policy,
        )
    return restored


def restore_revision_semantics_from_checkpoint(
    checkpoint: DecisionCheckpointState[T],
) -> tuple[DecisionRevisionState | None, DecisionRevisionPolicy | None]:
    """Return revision state and authoritative policy from one checkpoint."""
    restored = restore_decision_checkpoint_state(checkpoint)
    if restored.revision is None:
        return None, None
    return (
        revision_state_from_checkpoint(restored.revision),
        revision_policy_from_checkpoint(restored.revision),
    )


def _advance_lifecycle_to_terminal(
    lifecycle: DecisionLifecycleState,
) -> DecisionLifecycleState:
    """Advance one lifecycle state to terminal when durable outcome exists."""
    stage = lifecycle.stage
    if stage is DecisionLifecycleStage.TERMINAL:
        return lifecycle
    if stage is DecisionLifecycleStage.FINALIZATION:
        return transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)
    if stage is DecisionLifecycleStage.RESOLUTION:
        lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.FINALIZATION)
        return transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)
    if stage is DecisionLifecycleStage.REVISION:
        lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.RESOLUTION)
        lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.FINALIZATION)
        return transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)
    if stage is DecisionLifecycleStage.VERIFICATION:
        lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.RESOLUTION)
        lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.FINALIZATION)
        return transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)
    lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.VERIFICATION)
    lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.RESOLUTION)
    lifecycle = transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.FINALIZATION)
    return transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)


def reconcile_checkpoint_with_durable_finalization(
    checkpoint: DecisionCheckpointState[T],
    *,
    durable_guard: DecisionFinalizeGuardState[T],
) -> DecisionCheckpointState[T]:
    """Converge checkpoint finalization with durable authoritative outcome."""
    restored = restore_decision_checkpoint_state(checkpoint)
    key = decision_finalization_key(restored.lifecycle.identity)
    if durable_guard.key != key:
        raise ValueError("durable finalization key does not match checkpoint lifecycle")
    durable_outcome = durable_guard.authoritative_outcome
    if durable_outcome is None:
        return restored
    checkpoint_outcome = restored.finalization.authoritative_outcome
    if checkpoint_outcome is not None and checkpoint_outcome != durable_outcome:
        raise DecisionCheckpointCorruptionError(
            "checkpoint finalization conflicts with durable authoritative outcome",
        )
    lifecycle = restored.lifecycle
    if lifecycle.stage is not DecisionLifecycleStage.TERMINAL:
        lifecycle = _advance_lifecycle_to_terminal(lifecycle)
    return decision_checkpoint_state(
        lifecycle=lifecycle,
        finalization=durable_guard,
        revision=restored.revision,
    )


def load_resumable_decision_checkpoint(
    checkpoint_persistence: DecisionCheckpointPersistence[T],
    *,
    key: DecisionFinalizationKey,
    runtime_revision_policy: DecisionRevisionPolicy | None = None,
) -> DecisionCheckpointState[T] | None:
    """Load and validate one resumable checkpoint or return ``None`` when absent."""
    loaded = load_decision_checkpoint(checkpoint_persistence, key=key)
    if loaded is None:
        return None
    return resume_decision_checkpoint_state(
        loaded,
        runtime_revision_policy=runtime_revision_policy,
    )


def resume_decision_from_durable_state(
    *,
    checkpoint_persistence: DecisionCheckpointPersistence[T],
    finalization_persistence: DecisionFinalizationPersistence[T],
    key: DecisionFinalizationKey,
    runtime_revision_policy: DecisionRevisionPolicy | None = None,
) -> DecisionCheckpointState[T] | None:
    """Resume one decision from durable checkpoint and finalization stores."""
    durable_guard = load_decision_finalization_guard_state(
        finalization_persistence,
        key=key,
    )
    checkpoint = load_resumable_decision_checkpoint(
        checkpoint_persistence,
        key=key,
        runtime_revision_policy=runtime_revision_policy,
    )
    if checkpoint is None and durable_guard.authoritative_outcome is None:
        return None
    if checkpoint is None:
        outcome = durable_guard.authoritative_outcome
        if outcome is None:
            return None
        identity = outcome.identity
        lifecycle = _advance_lifecycle_to_terminal(
            initial_decision_lifecycle_state(identity),
        )
        checkpoint = decision_checkpoint_state(
            lifecycle=lifecycle,
            finalization=durable_guard,
        )
    if durable_guard.authoritative_outcome is not None:
        return reconcile_checkpoint_with_durable_finalization(
            checkpoint,
            durable_guard=durable_guard,
        )
    return checkpoint


def persist_terminal_decision_state(
    *,
    checkpoint_persistence: DecisionCheckpointPersistence[T],
    finalization_persistence: DecisionFinalizationPersistence[T],
    checkpoint: DecisionCheckpointState[T],
) -> DecisionCheckpointState[T]:
    """Commit durable outcome first, then persist terminal checkpoint."""
    restored = restore_decision_checkpoint_state(checkpoint)
    outcome = restored.finalization.authoritative_outcome
    if outcome is None:
        raise ValueError("terminal persistence requires authoritative outcome")
    key = restored.finalization.key
    commit_result = finalization_persistence.commit_authoritative_outcome(
        key=key,
        requested_outcome=outcome,
    )
    terminal_lifecycle = restored.lifecycle
    if terminal_lifecycle.stage is not DecisionLifecycleStage.TERMINAL:
        terminal_lifecycle = transition_decision_lifecycle(
            terminal_lifecycle,
            DecisionLifecycleStage.TERMINAL,
        )
    terminal_checkpoint = decision_checkpoint_state(
        lifecycle=terminal_lifecycle,
        finalization=commit_result.guard_state,
        revision=restored.revision,
    )
    save_decision_checkpoint(checkpoint_persistence, checkpoint=terminal_checkpoint)
    return terminal_checkpoint


def is_terminal_resumable_checkpoint(
    checkpoint: DecisionCheckpointState[T],
) -> bool:
    """Return whether checkpoint represents a finalized terminal decision."""
    restored = restore_decision_checkpoint_state(checkpoint)
    return (
        restored.lifecycle.stage is DecisionLifecycleStage.TERMINAL
        and restored.finalization.authoritative_outcome is not None
    )
