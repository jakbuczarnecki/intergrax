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
    """Provider-neutral continuation snapshot store with compare-and-swap semantics.

    Each exact four-ID Execution may retain many historical continuation episodes
    (distinct ``continuation_id`` values). At most one episode is **current** for
    that identity at any time. The current episode remains authoritative after
    terminal transition until a legal successor episode begins atomically.
    """

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
        """Return a snapshot only when exactly one record exists for the four-ID, else ``None``.

        Does not resolve the canonical **current** episode when multiple historical
        records exist; use :meth:`resolve_current_episode_for_identity` instead.
        """

    @abstractmethod
    def resolve_current_episode_for_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        """
        Return the canonical current continuation episode for exact four-ID identity.

        Returns ``None`` when no episode has ever been established. Raises
        :class:`ExecutionContinuationError` with ``AMBIGUOUS_IDENTITY`` when store
        state violates the single-current-episode invariant (fail closed).
        """

    @abstractmethod
    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        """
        Progress-gate lookup: canonical current episode for exact four-ID identity.

        Returns ``None`` when no current episode exists. Raises
        :class:`ExecutionContinuationError` with ``AMBIGUOUS_IDENTITY`` when the
        current episode cannot be resolved (fail closed).
        """

    @abstractmethod
    def begin_current_episode_if_predecessor_allows(
        self,
        pending: PendingExecutionContinuation,
    ) -> bool:
        """
        Atomically begin ``pending`` as the current episode for its four-ID identity.

        Returns ``False`` when ``continuation_id`` is already present. Raises
        :class:`ExecutionContinuationError` when the prior current episode is not
        terminal or when concurrent successor creation leaves an illegal state.
        """

    @abstractmethod
    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        """Persist ``pending`` only when ``continuation_id`` is not yet present.

        Does not establish current-episode succession; prefer
        :meth:`begin_current_episode_if_predecessor_allows` for new episodes.
        """

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
