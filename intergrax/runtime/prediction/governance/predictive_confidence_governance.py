# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Governed confidence composition (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive_risk import PredictiveRiskSignal


def compose_governed_confidence(
    *,
    raw_confidence: float,
    context_reliability: float,
    analyzer_precision: float,
    evidence_completeness: float,
) -> float:
    """
    confidence =
        raw_evidence
        × context reliability
        × analyzer historical precision
        × evidence completeness
    """
    raw = min(1.0, max(0.0, raw_confidence))
    ctx = min(1.0, max(0.0, context_reliability))
    analyzer = min(1.0, max(0.0, analyzer_precision))
    evidence = min(1.0, max(0.0, evidence_completeness))
    return min(1.0, max(0.0, raw * ctx * analyzer * evidence))


def evidence_completeness_for_signal(signal: PredictiveRiskSignal) -> float:
    if not signal.evidence_refs:
        return 0.0
    return min(1.0, 0.55 + 0.15 * len(signal.evidence_refs))


def govern_risk_signal_confidence(
    signal: PredictiveRiskSignal,
    *,
    context_quality: PredictiveContextQualityReport,
    analyzer_profile: PredictiveAnalyzerQualityProfile,
) -> PredictiveRiskSignal:
    governed = compose_governed_confidence(
        raw_confidence=signal.confidence,
        context_reliability=context_quality.reliability,
        analyzer_precision=analyzer_profile.precision,
        evidence_completeness=evidence_completeness_for_signal(signal),
    )
    return replace(signal, confidence=governed)


__all__ = [
    "compose_governed_confidence",
    "evidence_completeness_for_signal",
    "govern_risk_signal_confidence",
]
