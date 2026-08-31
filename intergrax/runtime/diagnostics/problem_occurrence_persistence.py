# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed persistence contract for durable ProblemOccurrence history (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intergrax.runtime.diagnostics.problem_lifecycle import ProblemId, ProblemOccurrence


class ProblemOccurrenceAppendResult(StrEnum):
    """Outcome of idempotent durable occurrence append."""

    CREATED = "created"
    ALREADY_EXISTS = "already_exists"


@dataclass(frozen=True, slots=True)
class ProblemOccurrencePage:
    """Bounded page of durable occurrences for one Problem."""

    items: tuple[ProblemOccurrence, ...]
    next_cursor: str | None
    has_more: bool


@dataclass(frozen=True, slots=True)
class ProblemOccurrenceAggregateStats:
    """Derived durable occurrence statistics for aggregate convergence."""

    occurrence_count: int
    first_seen_at: datetime
    last_seen_at: datetime


class ProblemOccurrencePersistenceIntegrityError(Exception):
    """Raised when stored occurrence data is malformed or inconsistent."""


class ProblemOccurrencePersistence(ABC):
    """
    Durable full history for accepted ProblemOccurrence records.

    Implementations must not require callers to load full history into memory.
    """

    @abstractmethod
    def append_if_absent(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        occurrence: ProblemOccurrence,
    ) -> ProblemOccurrenceAppendResult:
        """
        Persist one occurrence when absent.

        The same occurrence appended repeatedly must leave exactly one durable row.
        """

    @abstractmethod
    def query_occurrences(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
        limit: int,
        cursor: str | None = None,
    ) -> ProblemOccurrencePage:
        """
        Return one bounded page ordered by ``observed_at`` descending with
        deterministic occurrence-id tie-break.
        """

    @abstractmethod
    def aggregate_stats(
        self,
        *,
        tenant_id: str,
        problem_id: ProblemId,
    ) -> ProblemOccurrenceAggregateStats | None:
        """Return durable aggregate stats or ``None`` when no occurrences exist."""
