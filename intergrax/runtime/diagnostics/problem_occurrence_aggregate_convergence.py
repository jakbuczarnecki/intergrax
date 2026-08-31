# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Hot-path Problem aggregate deltas from accepted occurrences (DIAG-ENTERPRISE-2-R4)."""

from __future__ import annotations

from intergrax.runtime.diagnostics.problem_lifecycle import (
    Problem,
    ProblemLifecycleProvenance,
    ProblemOccurrence,
    ProblemStatus,
)
from intergrax.runtime.diagnostics.problem_lifecycle import (
    ProblemOccurrenceAggregateHealth,
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
        occurrence_aggregate_health=ProblemOccurrenceAggregateHealth.CONSISTENT,
    )
