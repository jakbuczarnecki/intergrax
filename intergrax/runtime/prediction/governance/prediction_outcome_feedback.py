# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Map diagnostic history outcomes to governance evaluations."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)
from intergrax.contracts.predictive.outcome_evaluation import (
    PredictionOutcomeEvaluation,
    build_outcome_evaluation,
)
from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)


def evaluation_from_history_outcome(
    record: PredictiveRiskHistoryRecord,
    *,
    evaluated_at: datetime | None = None,
) -> PredictionOutcomeEvaluation:
    when = evaluated_at or datetime.now(tz=UTC)
    outcome_type, status = _map_status(record.outcome_status)
    return PredictionOutcomeEvaluation(
        prediction_run_id=record.prediction_run_id,
        prediction_signal_id=record.risk_signal_id,
        tenant_id=record.tenant_id,
        analyzer_id=record.analyzer_id,
        risk_type=record.risk_type,
        predicted_at=record.generated_at,
        outcome_type=outcome_type,
        evaluation_status=status,
        evaluated_at=when,
        evidence_refs=record.evidence_refs,
        rationale=f"outcome_status={record.outcome_status.value}",
    )


def _map_status(
    status: PredictiveHistoryOutcomeStatus,
) -> tuple[PredictionOutcomeType, PredictionOutcomeEvaluationStatus]:
    if status is PredictiveHistoryOutcomeStatus.CONFIRMED:
        return (
            PredictionOutcomeType.TRUE_POSITIVE,
            PredictionOutcomeEvaluationStatus.EVALUATED,
        )
    if status is PredictiveHistoryOutcomeStatus.FALSE_POSITIVE:
        return (
            PredictionOutcomeType.FALSE_POSITIVE,
            PredictionOutcomeEvaluationStatus.EVALUATED,
        )
    if status is PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE:
        return (
            PredictionOutcomeType.UNKNOWN,
            PredictionOutcomeEvaluationStatus.INSUFFICIENT_EVIDENCE,
        )
    return (PredictionOutcomeType.UNKNOWN, PredictionOutcomeEvaluationStatus.UNKNOWN)


__all__ = ["evaluation_from_history_outcome", "build_outcome_evaluation"]
