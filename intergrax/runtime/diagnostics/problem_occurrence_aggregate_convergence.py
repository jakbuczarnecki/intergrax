# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded Problem aggregate convergence from durable occurrence history (DIAG-ENTERPRISE-2)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleProvenance,
    ProblemOccurrence,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_occurrence_persistence import (
    ProblemOccurrenceAggregateStats,
    ProblemOccurrencePersistence,
)


def apply_occurrence_delta_to_problem(
    existing: Problem,
    *,
    newly_accepted: tuple[ProblemOccurrence, ...],
    provenance: ProblemLifecycleProvenance,
) -> Problem:
    """Apply bounded aggregate deltas for occurrences accepted in this invocation."""
    if not newly_accepted:
        return existing

    observed_times = [occurrence.observed_at for occurrence in newly_accepted]
    next_status = existing.status
    if existing.status is ProblemStatus.RESOLVED:
        next_status = ProblemStatus.OPEN

    return Problem(
        problem_id=existing.problem_id,
        tenant_id=existing.tenant_id,
        status=next_status,
        first_seen_at=min(existing.first_seen_at, min(observed_times)),
        last_seen_at=max(existing.last_seen_at, max(observed_times)),
        occurrence_count=existing.occurrence_count + len(newly_accepted),
        provenance=provenance,
        record_version=existing.record_version + 1,
    )


def converge_problem_from_durable_stats(
    existing: Problem,
    *,
    stats: ProblemOccurrenceAggregateStats,
    provenance: ProblemLifecycleProvenance | None = None,
) -> Problem:
    """
    Converge aggregate counters and seen bounds from durable occurrence stats.

    Used after partial-write recovery when occurrence rows are authoritative.
    """
    next_status = existing.status
    if (
        existing.status is ProblemStatus.RESOLVED
        and stats.occurrence_count > existing.occurrence_count
    ):
        next_status = ProblemStatus.OPEN

    changed = (
        existing.occurrence_count != stats.occurrence_count
        or existing.first_seen_at != stats.first_seen_at
        or existing.last_seen_at != stats.last_seen_at
        or existing.status != next_status
    )
    if not changed:
        return existing

    return Problem(
        problem_id=existing.problem_id,
        tenant_id=existing.tenant_id,
        status=next_status,
        first_seen_at=stats.first_seen_at,
        last_seen_at=stats.last_seen_at,
        occurrence_count=stats.occurrence_count,
        provenance=provenance or existing.provenance,
        record_version=existing.record_version + 1,
    )


def problem_needs_occurrence_convergence(
    existing: Problem,
    *,
    occurrence_persistence: ProblemOccurrencePersistence,
    tenant_id: str,
) -> bool:
    stats = occurrence_persistence.aggregate_stats(
        tenant_id=tenant_id,
        problem_id=existing.problem_id,
    )
    if stats is None:
        return existing.occurrence_count != 0
    return (
        stats.occurrence_count != existing.occurrence_count
        or stats.first_seen_at != existing.first_seen_at
        or stats.last_seen_at != existing.last_seen_at
    )
