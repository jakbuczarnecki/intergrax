# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prediction outcome feedback contracts (PREDICTIVE R4 governance, R5 extended)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from intergrax.contracts.predictive.outcome.evaluation import (
    PredictionOutcomeEvaluation as _PredictionOutcomeEvaluation,
)
from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)

PredictionOutcomeEvaluation = _PredictionOutcomeEvaluation


class PredictionOutcomeEvaluationResult(StrEnum):
    TRUE_POSITIVE = "TRUE_POSITIVE"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    UNKNOWN = "UNKNOWN"


def _result_to_outcome_type(
    result: PredictionOutcomeEvaluationResult,
) -> PredictionOutcomeType:
    if result is PredictionOutcomeEvaluationResult.TRUE_POSITIVE:
        return PredictionOutcomeType.TRUE_POSITIVE
    if result is PredictionOutcomeEvaluationResult.FALSE_POSITIVE:
        return PredictionOutcomeType.FALSE_POSITIVE
    return PredictionOutcomeType.UNKNOWN


def build_outcome_evaluation(
    *,
    prediction_run_id: str,
    signal_id: str,
    tenant_id: str,
    analyzer_id: str,
    risk_type: str,
    predicted_at: datetime,
    evaluated_at: datetime,
    result: PredictionOutcomeEvaluationResult,
    rationale: str,
    evidence_refs: tuple[str, ...] = (),
) -> PredictionOutcomeEvaluation:
    """R4-compatible factory — maps legacy result labels to R5 outcome types."""
    outcome_type = _result_to_outcome_type(result)
    status = (
        PredictionOutcomeEvaluationStatus.EVALUATED
        if outcome_type is not PredictionOutcomeType.UNKNOWN
        else PredictionOutcomeEvaluationStatus.UNKNOWN
    )
    return PredictionOutcomeEvaluation(
        prediction_run_id=prediction_run_id,
        prediction_signal_id=signal_id,
        tenant_id=tenant_id,
        analyzer_id=analyzer_id,
        risk_type=risk_type,
        predicted_at=predicted_at,
        outcome_type=outcome_type,
        evaluation_status=status,
        evaluated_at=evaluated_at,
        evidence_refs=evidence_refs,
        rationale=rationale,
    )


__all__ = [
    "PredictionOutcomeEvaluation",
    "PredictionOutcomeEvaluationResult",
    "build_outcome_evaluation",
]
