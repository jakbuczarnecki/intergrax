# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Build readonly PredictiveContext from diagnostic read facts (PREDICTIVE R1)."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

from intergrax.contracts.predictive_context import (
    HistoricalProblemRef,
    PredictiveContext,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
)


def build_predictive_context_for_investigation(
    *,
    problem_detail: DiagnosticProblemDetail,
    occurrence: DiagnosticProblemOccurrenceView,
    extra_state: tuple[str, ...] = (),
) -> PredictiveContext:
    """Compose a bounded context snapshot — no Problem mutation."""

    tenant_id = problem_detail.tenant_id
    historical = tuple(
        HistoricalProblemRef(
            problem_id=str(problem_detail.problem_id),
            observed_at=occurrence.observed_at,
            summary="current_investigation_subject",
        ),
    )
    decision_history: tuple[str, ...] = ()
    if occurrence.decision_context is not None:
        decision_history = tuple(
            str(entry.decision_id)
            for entry in occurrence.decision_context.related_decisions
        )

    lineage_patterns: tuple[str, ...] = ()
    if occurrence.execution_lineage is not None:
        lineage_patterns = tuple(
            f"attempt:{attempt.attempt_id}"
            for attempt in occurrence.execution_lineage.attempts
        )

    snapshot_seed = (
        f"{tenant_id}|{problem_detail.problem_id}|{occurrence.observed_at.isoformat()}"
    )
    snapshot_id = f"pctx_{sha256(snapshot_seed.encode()).hexdigest()[:24]}"

    return PredictiveContext(
        tenant_id=tenant_id,
        current_state=(
            f"problem_status:{problem_detail.status.value}",
            f"occurrence_read:{occurrence.read_status.value}",
        )
        + extra_state,
        historical_problems=historical,
        execution_patterns=(),
        failure_history=(),
        performance_history=(),
        decision_history=decision_history,
        lineage_patterns=lineage_patterns,
        input_snapshot_id=snapshot_id or f"pctx_{uuid4().hex}",
        as_of=datetime.now(tz=UTC),
    )


__all__ = ["build_predictive_context_for_investigation"]
