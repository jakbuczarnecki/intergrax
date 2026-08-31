# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded Problem aggregate reconciliation from durable occurrence history (DIAG-ENTERPRISE-2-R4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemId,
    ProblemLifecycleIntegrityError,
    ProblemLifecycleProvenance,
    ProblemOccurrenceAggregateHealth,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrencePersistence,
)
from intergrax.runtime.diagnostics.problem_persistence import (
    ProblemPersistence,
    ProblemPersistenceConflictError,
)

DEFAULT_REPAIR_PAGE_SIZE = 500
_MAX_RECONCILIATION_ROUNDS = 32


@dataclass(frozen=True, slots=True)
class OccurrenceAggregateScan:
    """Exact aggregate derived from one paginated occurrence history scan."""

    occurrence_count: int
    first_seen_at: datetime | None
    last_seen_at: datetime | None


@dataclass(frozen=True, slots=True)
class OccurrenceAggregateAccumulator:
    """O(1) repair accumulator over paginated occurrence pages."""

    count: int = 0
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None


def accumulate_occurrence(
    accumulator: OccurrenceAggregateAccumulator,
    *,
    observed_at: datetime,
) -> OccurrenceAggregateAccumulator:
    if accumulator.count == 0:
        return OccurrenceAggregateAccumulator(
            count=1,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
        )
    return OccurrenceAggregateAccumulator(
        count=accumulator.count + 1,
        first_seen_at=min(accumulator.first_seen_at, observed_at),  # type: ignore[type-var]
        last_seen_at=max(accumulator.last_seen_at, observed_at),  # type: ignore[type-var]
    )


def scan_occurrence_aggregate(
    occurrence_persistence: ProblemOccurrencePersistence,
    *,
    tenant_id: str,
    problem_id: ProblemId,
    page_size: int = DEFAULT_REPAIR_PAGE_SIZE,
) -> OccurrenceAggregateScan:
    """Paginated authoritative scan — never loads full history into memory."""
    if type(page_size) is not int or isinstance(page_size, bool) or page_size < 1:
        raise ValueError("page_size must be a positive int")

    accumulator = OccurrenceAggregateAccumulator()
    cursor: str | None = None
    while True:
        page = occurrence_persistence.query_occurrences(
            tenant_id=tenant_id,
            problem_id=problem_id,
            limit=page_size,
            cursor=cursor,
        )
        for occurrence in page.items:
            accumulator = accumulate_occurrence(
                accumulator,
                observed_at=occurrence.observed_at,
            )
        if not page.has_more:
            break
        cursor = page.next_cursor

    return OccurrenceAggregateScan(
        occurrence_count=accumulator.count,
        first_seen_at=accumulator.first_seen_at,
        last_seen_at=accumulator.last_seen_at,
    )


def mark_problem_reconciliation_required(existing: Problem) -> Problem:
    if existing.occurrence_aggregate_health is ProblemOccurrenceAggregateHealth.RECONCILIATION_REQUIRED:
        return existing
    return Problem(
        problem_id=existing.problem_id,
        tenant_id=existing.tenant_id,
        status=existing.status,
        first_seen_at=existing.first_seen_at,
        last_seen_at=existing.last_seen_at,
        occurrence_count=existing.occurrence_count,
        provenance=existing.provenance,
        record_version=existing.record_version + 1,
        occurrence_aggregate_health=ProblemOccurrenceAggregateHealth.RECONCILIATION_REQUIRED,
    )


def converge_problem_from_occurrence_scan(
    existing: Problem,
    scan: OccurrenceAggregateScan,
    *,
    provenance: ProblemLifecycleProvenance | None = None,
) -> Problem:
    if scan.occurrence_count < 1 or scan.first_seen_at is None or scan.last_seen_at is None:
        raise ProblemLifecycleIntegrityError(
            "Problem occurrence aggregate reconciliation requires durable occurrence history",
        )

    next_status = existing.status
    if (
        existing.status is ProblemStatus.RESOLVED
        and scan.occurrence_count > existing.occurrence_count
    ):
        next_status = ProblemStatus.OPEN

    return Problem(
        problem_id=existing.problem_id,
        tenant_id=existing.tenant_id,
        status=next_status,
        first_seen_at=scan.first_seen_at,
        last_seen_at=scan.last_seen_at,
        occurrence_count=scan.occurrence_count,
        provenance=provenance or existing.provenance,
        record_version=existing.record_version + 1,
        occurrence_aggregate_health=ProblemOccurrenceAggregateHealth.CONSISTENT,
    )


def aggregate_matches_problem(existing: Problem, scan: OccurrenceAggregateScan) -> bool:
    if scan.first_seen_at is None or scan.last_seen_at is None:
        return existing.occurrence_count == 0
    return (
        existing.occurrence_count == scan.occurrence_count
        and existing.first_seen_at == scan.first_seen_at
        and existing.last_seen_at == scan.last_seen_at
    )


def reconcile_problem_occurrence_aggregate(
    existing: Problem,
    *,
    occurrence_persistence: ProblemOccurrencePersistence,
    problem_persistence: ProblemPersistence,
    page_size: int = DEFAULT_REPAIR_PAGE_SIZE,
    provenance: ProblemLifecycleProvenance | None = None,
) -> Problem:
    """
    Repeat-until-stable paginated repair.

    Hot-path updates may advance the Problem aggregate while scan is in flight;
    when ``Problem.occurrence_count`` exceeds the scan total the scan is repeated.
    """
    current = existing
    for _ in range(_MAX_RECONCILIATION_ROUNDS):
        scan = scan_occurrence_aggregate(
            occurrence_persistence,
            tenant_id=current.tenant_id,
            problem_id=current.problem_id,
            page_size=page_size,
        )
        latest = problem_persistence.get(
            tenant_id=current.tenant_id,
            problem_id=current.problem_id,
        )
        if latest is not None:
            current = latest

        if current.occurrence_count > scan.occurrence_count:
            continue

        if (
            current.occurrence_aggregate_health
            is ProblemOccurrenceAggregateHealth.CONSISTENT
            and aggregate_matches_problem(current, scan)
        ):
            return current

        next_record = converge_problem_from_occurrence_scan(
            current,
            scan,
            provenance=provenance,
        )
        if next_record == current:
            return current

        try:
            return problem_persistence.update(
                next_record,
                expected_version=current.record_version,
            )
        except ProblemPersistenceConflictError:
            refreshed = problem_persistence.get(
                tenant_id=current.tenant_id,
                problem_id=current.problem_id,
            )
            if refreshed is None:
                raise ProblemLifecycleIntegrityError(
                    "Problem disappeared during occurrence aggregate reconciliation",
                ) from None
            current = refreshed

    raise ProblemLifecycleIntegrityError(
        "occurrence aggregate reconciliation did not stabilize within bounded rounds",
    )
