# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-hosted Decision checkpoint durability port (DS-CORE-06).

Execution owns durable hosting; Decision contracts own semantic snapshot shape.
No storage backend or runtime wiring in this slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from intergrax.contracts.decision_checkpoint import (
    DecisionCheckpointState,
    restore_decision_checkpoint_state,
)
from intergrax.contracts.decision_finalization import DecisionFinalizationKey

T = TypeVar("T")


class StaleDecisionCheckpointWriteError(RuntimeError):
    """Materialized decision checkpoint projection conflict on revision CAS."""


@dataclass(frozen=True, slots=True)
class MaterializedDecisionCheckpoint(Generic[T]):
    """Read-for-update envelope: semantic snapshot and its concurrency token."""

    key: DecisionFinalizationKey
    checkpoint: DecisionCheckpointState[T]
    snapshot_revision: int

    def __post_init__(self) -> None:
        if self.snapshot_revision < 1:
            raise ValueError(
                "snapshot_revision must be >= 1 for a materialized checkpoint",
            )
        if self.key != self.checkpoint.finalization.key:
            raise ValueError(
                "materialized key must match checkpoint finalization key",
            )


class DecisionCheckpointPersistence(Protocol[T]):
    """Execution-facing durability port keyed by stable finalization scope."""

    def load(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> DecisionCheckpointState[T] | None:
        """Return a validated checkpoint or ``None`` when absent."""

    def load_materialized(
        self,
        *,
        key: DecisionFinalizationKey,
    ) -> MaterializedDecisionCheckpoint[T] | None:
        """Return checkpoint and revision token from one consistent store read."""

    def save(
        self,
        *,
        checkpoint: DecisionCheckpointState[T],
        expected_revision: int | None = None,
    ) -> None:
        """Persist one validated checkpoint snapshot.

        When ``expected_revision`` is set, the store must reject concurrent writers with
        :class:`StaleDecisionCheckpointWriteError` if the materialized revision changed.
        """


def load_materialized_decision_checkpoint(
    persistence: DecisionCheckpointPersistence[T],
    *,
    key: DecisionFinalizationKey,
) -> MaterializedDecisionCheckpoint[T] | None:
    """Load, validate, and return the materialized read-for-update envelope."""
    loaded = persistence.load_materialized(key=key)
    if loaded is None:
        return None
    validated_checkpoint = restore_decision_checkpoint_state(loaded.checkpoint)
    if validated_checkpoint is loaded.checkpoint:
        return loaded
    return MaterializedDecisionCheckpoint(
        key=loaded.key,
        checkpoint=validated_checkpoint,
        snapshot_revision=loaded.snapshot_revision,
    )


def load_decision_checkpoint(
    persistence: DecisionCheckpointPersistence[T],
    *,
    key: DecisionFinalizationKey,
) -> DecisionCheckpointState[T] | None:
    """Load and validate a checkpoint from Execution-hosted durability."""
    materialized = load_materialized_decision_checkpoint(persistence, key=key)
    if materialized is None:
        return None
    return materialized.checkpoint


def save_decision_checkpoint(
    persistence: DecisionCheckpointPersistence[T],
    *,
    checkpoint: DecisionCheckpointState[T],
    expected_revision: int | None = None,
) -> None:
    """Validate and persist one checkpoint snapshot."""
    validated = restore_decision_checkpoint_state(checkpoint)
    persistence.save(checkpoint=validated, expected_revision=expected_revision)
