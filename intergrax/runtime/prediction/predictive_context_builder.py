# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Build readonly PredictiveContext from diagnostic read facts (PREDICTIVE R1/R4)."""

from __future__ import annotations

from intergrax.contracts.predictive import (
    HistoricalProblemRef,
    PredictiveContext,
    PredictiveContextDiagnostic,
    PredictiveScope,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
)
from intergrax.runtime.prediction.context import (
    DecisionHistoryProvider,
    DiagnosticHistoryProvider,
    PredictiveContextAggregator,
    PredictiveContextBuilder,
)


def build_predictive_context_for_investigation(
    *,
    problem_detail: DiagnosticProblemDetail,
    occurrence: DiagnosticProblemOccurrenceView,
    extra_state: tuple[str, ...] = (),
) -> PredictiveContext:
    """Compose a bounded context snapshot — no Problem mutation."""

    tenant_id = problem_detail.tenant_id
    historical = (
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

    scope = PredictiveScope(tenant_id=tenant_id)
    diagnostic = PredictiveContextDiagnostic(historical_problems=historical)
    current_state = (
        f"problem_status:{problem_detail.status.value}",
        f"occurrence_read:{occurrence.read_status.value}",
    ) + extra_state

    builder = PredictiveContextBuilder(
        aggregator=PredictiveContextAggregator(
            providers=(
                DiagnosticHistoryProvider(
                    diagnostic=diagnostic,
                    lineage_patterns=lineage_patterns,
                    current_state=current_state,
                ),
                DecisionHistoryProvider(decision_history=decision_history),
            ),
        ),
    )
    return builder.build(scope)


__all__ = ["build_predictive_context_for_investigation"]
