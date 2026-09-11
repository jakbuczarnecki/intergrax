# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Build PredictiveQualityAssessment for audit and investigation reads."""

from __future__ import annotations

from intergrax.contracts.predictive.analyzer_quality_profile import PredictiveAnalyzerQualityProfile
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive.quality import PredictiveQualityAssessment, PredictiveQualityDimension
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction.governance.predictive_confidence_governance import (
    compose_governed_confidence,
    evidence_completeness_for_signal,
)


def assess_prediction_quality(
    *,
    context_quality: PredictiveContextQualityReport,
    analyzer_profile: PredictiveAnalyzerQualityProfile,
    signal: PredictiveRiskSignal,
) -> PredictiveQualityAssessment:
    evidence_score = evidence_completeness_for_signal(signal)
    governed = compose_governed_confidence(
        raw_confidence=signal.confidence,
        context_reliability=context_quality.reliability,
        analyzer_precision=analyzer_profile.precision,
        evidence_completeness=evidence_score,
    )
    explanation = build_explanation_lines(
        signal=signal,
        context_quality=context_quality,
        analyzer_profile=analyzer_profile,
    )
    return PredictiveQualityAssessment(
        context_quality=PredictiveQualityDimension(
            label="context_reliability",
            score=context_quality.reliability,
            rationale=(
                f"completeness={context_quality.completeness.value}; "
                f"coverage={context_quality.coverage:.2f}"
            ),
        ),
        analyzer_quality=PredictiveQualityDimension(
            label="analyzer_precision",
            score=analyzer_profile.precision,
            rationale=(
                f"historical precision {analyzer_profile.precision:.0%} "
                f"over {analyzer_profile.predictions} predictions"
            ),
        ),
        evidence_quality=PredictiveQualityDimension(
            label="evidence_completeness",
            score=evidence_score,
            rationale=f"{len(signal.evidence_refs)} evidence refs",
        ),
        confidence_quality=PredictiveQualityDimension(
            label="governed_confidence",
            score=governed,
            rationale="raw × context × analyzer × evidence",
        ),
        completeness=context_quality.completeness,
        governed_confidence=governed,
        explanation=explanation,
    )


def build_explanation_lines(
    *,
    signal: PredictiveRiskSignal,
    context_quality: PredictiveContextQualityReport,
    analyzer_profile: PredictiveAnalyzerQualityProfile,
) -> tuple[str, ...]:
    lines: list[str] = [signal.summary]
    if signal.risk_type:
        lines.append(f"risk_type={signal.risk_type}")
    if context_quality.missing_sections:
        lines.append(f"missing_context={','.join(context_quality.missing_sections)}")
    lines.append(f"analyzer precision {analyzer_profile.precision:.0%}")
    return tuple(lines)


__all__ = ["assess_prediction_quality", "build_explanation_lines"]
