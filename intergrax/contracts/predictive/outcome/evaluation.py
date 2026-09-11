# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prediction outcome evaluation record (PREDICTIVE R5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive.outcome.types import (
    PredictionOutcomeEvaluationStatus,
    PredictionOutcomeType,
)


@dataclass(frozen=True, slots=True)
class PredictionOutcomeEvaluation:
    """
    Closes the loop: prediction → evidence-backed outcome → quality measurement.

    Never mints Problems or mutates diagnostic truth.
    """

    prediction_run_id: str
    prediction_signal_id: str
    tenant_id: str
    analyzer_id: str
    outcome_type: PredictionOutcomeType
    evaluation_status: PredictionOutcomeEvaluationStatus
    evaluated_at: datetime
    evidence_refs: tuple[str, ...]
    risk_type: str = ""
    predicted_at: datetime | None = None
    rationale: str = ""
    resolver_id: str = ""
    evaluation_confidence: float | None = None

    @property
    def signal_id(self) -> str:
        """Backward-compatible alias (R4 governance)."""
        return self.prediction_signal_id

    def __post_init__(self) -> None:
        if not self.prediction_run_id.strip():
            raise ValueError("prediction_run_id must be non-empty")
        if not self.prediction_signal_id.strip():
            raise ValueError("prediction_signal_id must be non-empty")
        if not self.tenant_id.strip():
            raise ValueError("tenant_id must be non-empty")
        if not self.analyzer_id.strip():
            raise ValueError("analyzer_id must be non-empty")
        if self.evaluation_confidence is not None and not (
            0.0 <= self.evaluation_confidence <= 1.0
        ):
            raise ValueError("evaluation_confidence must be in [0.0, 1.0]")


def outcome_type_to_legacy_result(
    outcome_type: PredictionOutcomeType,
) -> str:
    """Maps R5 outcome types to R4-style labels for audit consumers."""
    if outcome_type is PredictionOutcomeType.TRUE_POSITIVE:
        return "TRUE_POSITIVE"
    if outcome_type is PredictionOutcomeType.FALSE_POSITIVE:
        return "FALSE_POSITIVE"
    return "UNKNOWN"


__all__ = ["PredictionOutcomeEvaluation", "outcome_type_to_legacy_result"]
