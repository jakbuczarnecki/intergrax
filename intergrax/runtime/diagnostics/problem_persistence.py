# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persistence contract for stable diagnostic Problem records (DIAG-5D)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from intergrax.runtime.diagnostics.problem_grouping import ProblemGroupingSubjectRef

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.problem_lifecycle import (
        Problem,
        ProblemId,
        ProblemReconciliationKey,
        ProblemStatus,
    )


@dataclass(frozen=True, slots=True)
class ProblemListPage:
    """Bounded page of canonical Problems in public list order."""

    problems: tuple[Problem, ...]
    next_cursor: str | None
    has_more: bool


class ProblemPersistenceConflictError(Exception):
    """Raised when a write conflicts with an existing record or CAS version."""


class ProblemPersistenceIntegrityReason(StrEnum):
    """Typed integrity failure reasons for transient or diagnostic classification."""

    RECONCILIATION_WINNER_CANONICAL_PENDING = (
        "reconciliation_winner_canonical_pending"
    )


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


class ProblemPersistence(ABC):
    """
    Durable store for derived diagnostic Problem records.

  Problems are persisted derived operational diagnostic state — not canonical
  execution truth. Implementations (SQLite, Postgres, …) live behind this
  contract.
    """

    @abstractmethod
    def get(self, *, tenant_id: str, problem_id: ProblemId) -> Problem | None:
        """Return one tenant-scoped Problem or ``None`` when absent."""

    @abstractmethod
    def list_for_tenant(self, tenant_id: str) -> tuple[Problem, ...]:
        """
        Return all Problems for a tenant in stable ``problem_id`` order.

        Legacy/testing helper — operator reads must use ``query_problems`` instead.
        """

    @abstractmethod
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

    @abstractmethod
    def find_by_reconciliation_key(
        self,
        *,
        tenant_id: str,
        reconciliation_key: ProblemReconciliationKey,
    ) -> Problem | None:
        """Return the Problem indexed by the typed reconciliation key, if any."""

    @abstractmethod
    def find_by_subject_ref(
        self,
        *,
        tenant_id: str,
        subject_ref: ProblemGroupingSubjectRef,
    ) -> Problem | None:
        """Return the Problem that already accepted ``subject_ref``, if any."""

    @abstractmethod
    def create(self, record: Problem) -> Problem:
        """
        Persist a new Problem atomically.

        Idempotent when the same ``problem_id`` is written with identical content.
        Raises ``ProblemPersistenceConflictError`` on identity or index conflicts.
        """

    @abstractmethod
    def update(self, record: Problem, *, expected_version: int) -> Problem:
        """
        Compare-and-set update for an existing Problem.

        Raises ``ProblemPersistenceConflictError`` when ``expected_version`` does
        not match the stored record version.
        """

    def close(self) -> None:
        """Release backend resources (no-op for most stores)."""
