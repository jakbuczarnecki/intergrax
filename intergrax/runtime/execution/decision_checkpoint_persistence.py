# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-hosted Decision checkpoint durability port (DS-CORE-06).

Execution owns durable hosting; Decision contracts own semantic snapshot shape.
No storage backend or runtime wiring in this slice.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    restore_decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import DecisionFinalizationKey

T = TypeVar("T")


class DecisionCheckpointPersistence(Protocol[T]):
    """Execution-facing durability port keyed by stable finalization scope."""

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[T] | None:
        """Return a validated checkpoint or ``None`` when absent."""

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[T],
    ) -> None:
        """Persist one validated checkpoint snapshot."""


def load_decision_checkpoint(
    persistence: DecisionCheckpointPersistence[T],
    *,
    key: DecisionFinalizationKey,
) -> DecisionCheckpointState[T] | None:
    """Load and validate a checkpoint from Execution-hosted durability."""
    loaded = persistence.load(key=key)
    if loaded is None:
        return None
    return restore_decision_checkpoint_state(loaded)


def save_decision_checkpoint(
    persistence: DecisionCheckpointPersistence[T],
    *,
    checkpoint: DecisionCheckpointState[T],
) -> None:
    """Validate and persist one checkpoint snapshot."""
    validated = restore_decision_checkpoint_state(checkpoint)
    persistence.save(checkpoint=validated)
