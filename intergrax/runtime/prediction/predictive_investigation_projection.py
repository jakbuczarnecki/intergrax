# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Map predictive engine output to investigation read models (PREDICTIVE R1)."""

from __future__ import annotations

from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive.quality import PredictiveQualityAssessment
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.contracts.predictive_investigation_read import RelatedPredictiveRiskSignalView

FORECAST_RISK_TYPES: frozenset[str] = frozenset(
    {
        "LATENCY_DEGRADATION",
        "FAILURE_ACCELERATION_RISK",
        "RETRY_STORM_RISK",
        "RESOURCE_EXHAUSTION_RISK",
    },
)


def project_related_risk_signals(
    signals: tuple[PredictiveRiskSignal, ...],
    *,
    analyzer_id_by_signal: dict[str, str] | None = None,
    quality_by_signal_id: dict[str, PredictiveQualityAssessment] | None = None,
    context_quality: PredictiveContextQualityReport | None = None,
) -> tuple[RelatedPredictiveRiskSignalView, ...]:
    mapping = analyzer_id_by_signal or {}
    qualities = quality_by_signal_id or {}
    views: list[RelatedPredictiveRiskSignalView] = []
    for signal in signals:
        assessment = qualities.get(signal.signal_id)
        explanation = assessment.explanation if assessment else (signal.summary,)
        views.append(
            RelatedPredictiveRiskSignalView(
                signal_id=signal.signal_id,
                tenant_id=signal.tenant_id,
                scope=signal.scope,
                subject_identity=signal.subject_identity,
                risk_type=signal.risk_type,
                severity=signal.severity,
                confidence=signal.confidence,
                evidence_refs=signal.evidence_refs,
                prediction_window_label=signal.prediction_window.label,
                generated_at=signal.generated_at,
                model_version=signal.model_version,
                summary=signal.summary,
                recommended_actions=signal.recommended_actions,
                analyzer_id=mapping.get(
                    signal.signal_id,
                    signal.analyzer_metadata.analyzer_id,
                ),
                prediction_quality=assessment,
                context_quality=context_quality,
                prediction_explanation=explanation,
            ),
        )
    return tuple(views)


def project_forecast_risk_signals(
    signals: tuple[PredictiveRiskSignal, ...],
    *,
    quality_by_signal_id: dict[str, PredictiveQualityAssessment] | None = None,
    context_quality: PredictiveContextQualityReport | None = None,
) -> tuple[RelatedPredictiveRiskSignalView, ...]:
    """Readonly R3 statistical forecasts for DiagnosticInvestigationView."""
    return tuple(
        view
        for view in project_related_risk_signals(
            signals,
            quality_by_signal_id=quality_by_signal_id,
            context_quality=context_quality,
        )
        if view.risk_type in FORECAST_RISK_TYPES
    )


__all__ = ["FORECAST_RISK_TYPES", "project_forecast_risk_signals", "project_related_risk_signals"]
