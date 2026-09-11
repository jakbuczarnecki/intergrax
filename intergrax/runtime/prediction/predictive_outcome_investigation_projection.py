# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Project outcome evaluations onto investigation read models (PREDICTIVE R5)."""

from __future__ import annotations

from intergrax.contracts.predictive.outcome.evaluation import PredictionOutcomeEvaluation
from intergrax.contracts.predictive_investigation_read import RelatedPredictionOutcomeHistoryView


def project_prediction_outcome_history(
    evaluations: tuple[PredictionOutcomeEvaluation, ...],
    *,
    precision_delta_by_signal: dict[str, str] | None = None,
) -> tuple[RelatedPredictionOutcomeHistoryView, ...]:
    deltas = precision_delta_by_signal or {}
    views: list[RelatedPredictionOutcomeHistoryView] = []
    for row in evaluations:
        views.append(
            RelatedPredictionOutcomeHistoryView(
                prediction_signal_id=row.prediction_signal_id,
                tenant_id=row.tenant_id,
                prediction_run_id=row.prediction_run_id,
                risk_type=row.risk_type,
                analyzer_id=row.analyzer_id,
                outcome_type=row.outcome_type,
                evaluation_status=row.evaluation_status,
                evaluated_at=row.evaluated_at,
                evidence_refs=row.evidence_refs,
                rationale=row.rationale,
                precision_delta_label=deltas.get(row.prediction_signal_id, ""),
            ),
        )
    return tuple(views)


__all__ = ["project_prediction_outcome_history"]
