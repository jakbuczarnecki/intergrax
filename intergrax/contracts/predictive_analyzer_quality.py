# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Analyzer effectiveness metrics derived from prediction history (PREDICTIVE R2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictiveAnalyzerQualityMetrics:
    """Quality view for one analyzer — counts terminal outcomes only."""

    analyzer_id: str
    prediction_count: int
    confirmed_count: int
    false_positive_count: int
    unknown_count: int
    precision: float
    coverage: float
    confidence_accuracy: float

    def __post_init__(self) -> None:
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if self.prediction_count < 0:
            raise ValueError("prediction_count must be non-negative")
        if not (0.0 <= self.precision <= 1.0):
            raise ValueError("precision must be in [0.0, 1.0]")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0.0, 1.0]")
        if not (0.0 <= self.confidence_accuracy <= 1.0):
            raise ValueError("confidence_accuracy must be in [0.0, 1.0]")


def compute_analyzer_quality_metrics(
    *,
    analyzer_id: str,
    records: tuple[object, ...],
) -> PredictiveAnalyzerQualityMetrics:
    """Aggregate metrics from ``PredictiveRiskHistoryRecord`` rows (duck-typed outcome)."""
    from intergrax.contracts.predictive_history import PredictiveHistoryOutcomeStatus

    prediction_count = len(records)
    confirmed = 0
    false_positive = 0
    unknown = 0
    evaluated_with_confidence = 0
    confidence_hit = 0.0

    for record in records:
        status = record.outcome_status
        if status is PredictiveHistoryOutcomeStatus.CONFIRMED:
            confirmed += 1
            evaluated_with_confidence += 1
            confidence_hit += float(record.confidence)
        elif status is PredictiveHistoryOutcomeStatus.FALSE_POSITIVE:
            false_positive += 1
            evaluated_with_confidence += 1
            confidence_hit += 1.0 - float(record.confidence)
        elif status in (
            PredictiveHistoryOutcomeStatus.UNKNOWN,
            PredictiveHistoryOutcomeStatus.INSUFFICIENT_EVIDENCE,
            PredictiveHistoryOutcomeStatus.EXPIRED,
        ):
            unknown += 1

    decided = confirmed + false_positive
    precision = (confirmed / prediction_count) if prediction_count else 0.0
    coverage = (decided / prediction_count) if prediction_count else 0.0
    confidence_accuracy = (
        confidence_hit / evaluated_with_confidence if evaluated_with_confidence else 0.0
    )

    return PredictiveAnalyzerQualityMetrics(
        analyzer_id=analyzer_id,
        prediction_count=prediction_count,
        confirmed_count=confirmed,
        false_positive_count=false_positive,
        unknown_count=unknown,
        precision=precision,
        coverage=coverage,
        confidence_accuracy=confidence_accuracy,
    )


__all__ = ["PredictiveAnalyzerQualityMetrics", "compute_analyzer_quality_metrics"]
