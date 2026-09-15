# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""GR-5-R2 — atomic persistence for canonical execution continuation snapshots."""

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationIdentity,
    PendingExecutionContinuation,
)


class ExecutionContinuationStateStore(ABC):
    """Provider-neutral continuation snapshot store with compare-and-swap semantics."""

    @property
    @abstractmethod
    def is_durable(self) -> bool:
        """Whether continuation snapshots survive process restart."""

    @abstractmethod
    def load(self, continuation_id: str) -> PendingExecutionContinuation | None:
        """Return the snapshot for ``continuation_id`` or ``None`` when absent."""

    @abstractmethod
    def find_by_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        """Return the unique snapshot for exact four-ID identity, else ``None``."""

    @abstractmethod
    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        """
        Progress-gate lookup for exact four-ID identity.

        Returns ``None`` when no continuation exists. Raises
        :class:`ExecutionContinuationError` with ``AMBIGUOUS_IDENTITY`` when more
        than one snapshot matches (fail closed).
        """

    @abstractmethod
    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        """Persist ``pending`` only when ``continuation_id`` is not yet present."""

    @abstractmethod
    def compare_and_swap(
        self,
        *,
        continuation_id: str,
        expected: PendingExecutionContinuation,
        updated: PendingExecutionContinuation,
    ) -> bool:
        """
        Atomically replace ``expected`` with ``updated`` when the stored snapshot
        equals ``expected`` (continuation id, revision, lifecycle, identity).
        """


__all__ = ["ExecutionContinuationStateStore"]
