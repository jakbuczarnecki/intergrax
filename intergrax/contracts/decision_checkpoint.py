# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision System checkpoint contracts (DS-CORE-06).

Typed durable snapshot of Decision lifecycle and finalization guard state for
canonical Execution-hosted persistence. Capture, validate, and restore only —
no lifecycle transitions, finalization logic, or storage backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from intergrax.contracts.decision_finalization import (
    DecisionFinalizeGuardState,
    decision_finalization_key,
)
from intergrax.contracts.decision_lifecycle import (
    DecisionLifecycleStage,
    DecisionLifecycleState,
)

T = TypeVar("T")

_POST_FINALIZATION_STAGES = frozenset(
    {
        DecisionLifecycleStage.FINALIZATION,
        DecisionLifecycleStage.TERMINAL,
    },
)


@dataclass(frozen=True, slots=True)
class DecisionCheckpointState(Generic[T]):
    """Immutable semantic checkpoint for one decision lifecycle position."""

    lifecycle: DecisionLifecycleState
    finalization: DecisionFinalizeGuardState[T]

    def __post_init__(self) -> None:
        if type(self.lifecycle) is not DecisionLifecycleState:
            raise TypeError("DecisionCheckpointState.lifecycle must be DecisionLifecycleState")
        if type(self.finalization) is not DecisionFinalizeGuardState:
            raise TypeError(
                "DecisionCheckpointState.finalization must be DecisionFinalizeGuardState",
            )
        validate_decision_checkpoint_state(self)


def validate_decision_checkpoint_state(
    checkpoint: DecisionCheckpointState[T],
) -> DecisionCheckpointState[T]:
    """Fail closed on identity, finalization key, or semantic inconsistencies."""
    if type(checkpoint) is not DecisionCheckpointState:
        raise TypeError("checkpoint must be DecisionCheckpointState")

    lifecycle = checkpoint.lifecycle
    finalization = checkpoint.finalization
    expected_key = decision_finalization_key(lifecycle.identity)
    if expected_key != finalization.key:
        raise ValueError(
            "decision checkpoint lifecycle identity does not match finalization key: "
            f"decision_id={lifecycle.identity.decision_id!r}, "
            f"tenant_id={lifecycle.identity.tenant_id!r}, "
            f"scope={lifecycle.identity.scope.namespace!r}/"
            f"{lifecycle.identity.scope.subject!r}",
        )

    stage = lifecycle.stage
    outcome = finalization.authoritative_outcome

    if outcome is not None and stage not in _POST_FINALIZATION_STAGES:
        raise ValueError(
            "decision checkpoint authoritative outcome requires stage "
            f"finalization or terminal, got {stage.value!r}",
        )

    if stage is DecisionLifecycleStage.TERMINAL and outcome is None:
        raise ValueError(
            "decision checkpoint terminal stage requires authoritative outcome",
        )

    return checkpoint


def decision_checkpoint_state(
    *,
    lifecycle: DecisionLifecycleState,
    finalization: DecisionFinalizeGuardState[T],
) -> DecisionCheckpointState[T]:
    """Capture one validated immutable checkpoint from current semantic state."""
    if type(lifecycle) is not DecisionLifecycleState:
        raise TypeError("lifecycle must be DecisionLifecycleState")
    if type(finalization) is not DecisionFinalizeGuardState:
        raise TypeError("finalization must be DecisionFinalizeGuardState")
    return DecisionCheckpointState(lifecycle=lifecycle, finalization=finalization)


def restore_decision_checkpoint_state(
    checkpoint: DecisionCheckpointState[T],
) -> DecisionCheckpointState[T]:
    """Restore exact checkpointed semantic state after durable load."""
    validate_decision_checkpoint_state(checkpoint)
    return checkpoint
