# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Diagnostic Problem persistence port — enterprise storage boundary (HARDENING-8)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.diagnostics.problem_identity import ProblemId, ProblemStatus
from intergrax.contracts.diagnostics.problem_record import PersistedProblem
from intergrax.contracts.diagnostics.reconciliation_key import ProblemReconciliationKey
from intergrax.contracts.diagnostics.subject_ref import ProblemGroupingSubjectRef


@dataclass(frozen=True, slots=True)
class ProblemListPage:
    """Bounded page of canonical Problems in public list order."""

    problems: tuple[PersistedProblem, ...]
    next_cursor: str | None
    has_more: bool


class ProblemPersistenceConflictError(Exception):
    """Raised when a write conflicts with an existing record or CAS version."""


class ProblemPersistenceIntegrityReason(StrEnum):
    """Typed integrity failure reasons for transient or diagnostic classification."""

    RECONCILIATION_WINNER_CANONICAL_PENDING = (
        "reconciliation_winner_canonical_pending"
    )
    LIST_INDEX_CANONICAL_METADATA_MISMATCH = "list_index_canonical_metadata_mismatch"


class ProblemPersistenceIntegrityError(Exception):
    """Raised when indexed storage is inconsistent with the canonical Problem record."""

    def __init__(
        self,
        message: str,
        *,
        reason: ProblemPersistenceIntegrityReason | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason


@runtime_checkable
class ProblemPersistence(Protocol):
    """
    Durable store for derived diagnostic Problem records.

    Problems are persisted derived operational diagnostic state — not canonical
    execution truth. Adapters (in-memory, document store, external providers)
    implement this port at the composition root.
    """

    def get(self, *, tenant_id: str, problem_id: ProblemId) -> PersistedProblem | None:
        """Return one tenant-scoped Problem or ``None`` when absent."""

    def query_problems(
        self,
        *,
        tenant_id: str,
        status: ProblemStatus | None = None,
        limit: int,
        cursor: str | None = None,
    ) -> ProblemListPage:
        """
        Return one bounded page of Problems ordered by ``last_seen_at`` descending
        with ``problem_id`` ascending tie-break.

        ``cursor`` continues a prior page for the same tenant and status filter.
        """

    def find_by_reconciliation_key(
        self,
        *,
        tenant_id: str,
        reconciliation_key: ProblemReconciliationKey,
    ) -> PersistedProblem | None:
        """Return the Problem indexed by the typed reconciliation key, if any."""

    def find_by_subject_ref(
        self,
        *,
        tenant_id: str,
        subject_ref: ProblemGroupingSubjectRef,
    ) -> PersistedProblem | None:
        """Return the Problem that already accepted ``subject_ref``, if any."""

    def create(
        self,
        record: PersistedProblem,
        *,
        indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
    ) -> PersistedProblem:
        """
        Persist a new Problem atomically.

        ``indexed_subject_refs`` seeds durable subject→Problem lookup indexes.
        Idempotent when the same ``problem_id`` is written with identical content.
        Raises ``ProblemPersistenceConflictError`` on identity or index conflicts.
        """

    def update(
        self,
        record: PersistedProblem,
        *,
        expected_version: int,
        indexed_subject_refs: tuple[ProblemGroupingSubjectRef, ...] = (),
    ) -> PersistedProblem:
        """
        Compare-and-set update for an existing Problem.

        Raises ``ProblemPersistenceConflictError`` when ``expected_version`` does
        not match the stored record version.
        """

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""


__all__ = [
    "ProblemListPage",
    "ProblemPersistence",
    "ProblemPersistenceConflictError",
    "ProblemPersistenceIntegrityError",
    "ProblemPersistenceIntegrityReason",
]
