# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Historical calibration for governed confidence (PREDICTIVE R5)."""

from __future__ import annotations

from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.runtime.prediction.governance.predictive_confidence_governance import (
    compose_governed_confidence,
    evidence_completeness_for_signal,
)
from intergrax.contracts.predictive_risk import PredictiveRiskSignal


def calibrate_analyzer_confidence(
    *,
    raw_confidence: float,
    analyzer_profile: PredictiveAnalyzerQualityProfile,
    context_quality: PredictiveContextQualityReport,
    signal: PredictiveRiskSignal,
) -> float:
    """
    raw confidence × historical calibration × context × evidence completeness.

    Example: raw 0.95 with precision/calibration ~0.68 → governed ~0.65 (with context factors).
    """
    return compose_governed_confidence(
        raw_confidence=raw_confidence,
        context_reliability=context_quality.reliability,
        analyzer_precision=analyzer_profile.confidence_calibration,
        evidence_completeness=evidence_completeness_for_signal(signal),
    )


class PredictiveConfidenceCalibrator:
    """Stateful-free calibrator — profiles supplied by governance store."""

    def calibrate_signal(
        self,
        signal: PredictiveRiskSignal,
        *,
        context_quality: PredictiveContextQualityReport,
        analyzer_profile: PredictiveAnalyzerQualityProfile,
    ) -> float:
        return calibrate_analyzer_confidence(
            raw_confidence=signal.confidence,
            analyzer_profile=analyzer_profile,
            context_quality=context_quality,
            signal=signal,
        )


__all__ = ["PredictiveConfidenceCalibrator", "calibrate_analyzer_confidence"]
