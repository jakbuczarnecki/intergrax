# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation confidence composition (PREVENTIVE R6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.contracts.preventive.context import HistoricalOutcome


def compose_preventive_confidence(
    *,
    risk_confidence: float,
    analyzer_quality: float,
    historical_success: float,
    context_completeness: float,
) -> float:
    """
    risk × analyzer quality × historical success × context completeness.

    Example: 0.9 × 0.8 × 0.7 × 0.9 ≈ 0.45
    """
    for name, value in (
        ("risk_confidence", risk_confidence),
        ("analyzer_quality", analyzer_quality),
        ("historical_success", historical_success),
        ("context_completeness", context_completeness),
    ):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0.0, 1.0]")
    product = risk_confidence * analyzer_quality * historical_success * context_completeness
    return min(1.0, max(0.0, product))


@dataclass(frozen=True, slots=True)
class PreventiveConfidenceEvaluator:
    def evaluate(
        self,
        *,
        risk_signal: PredictiveRiskSignal,
        analyzer_profile: PredictiveAnalyzerQualityProfile,
        historical_outcome: HistoricalOutcome,
        context_quality: PredictiveContextQualityReport,
    ) -> float:
        return compose_preventive_confidence(
            risk_confidence=risk_signal.confidence,
            analyzer_quality=analyzer_profile.confidence_calibration,
            historical_success=historical_outcome.historical_success_rate,
            context_completeness=(context_quality.coverage + context_quality.reliability) / 2.0,
        )


__all__ = ["PreventiveConfidenceEvaluator", "compose_preventive_confidence"]
