# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Shared forecast signal stamping (PREDICTIVE R3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from intergrax.contracts.forecast_analyzer_descriptor import ForecastAnalyzerDescriptor
from intergrax.contracts.predictive_feature_set import SubjectPredictiveFeatures
from intergrax.contracts.predictive_historical_intelligence import HistoricalRiskIntelligence
from intergrax.contracts.predictive_risk import (
    PREDICTION_RUN_ID_ENGINE_STAMP,
    PredictiveAnalyzerMetadata,
    PredictiveRiskScope,
    PredictiveRiskSeverity,
    PredictiveRiskSignal,
    PredictiveWindow,
    mint_predictive_signal_id,
)
from intergrax.runtime.prediction.forecasting.confidence_model import compose_forecast_confidence


def emit_forecast_signal(
    *,
    tenant_id: str,
    subject: SubjectPredictiveFeatures,
    descriptor: ForecastAnalyzerDescriptor,
    risk_type: str,
    severity: PredictiveRiskSeverity,
    evidence_strength: float,
    historical_intelligence: HistoricalRiskIntelligence,
    evidence_refs: tuple[str, ...],
    summary: str,
    window_minutes: int,
    window_label: str,
) -> PredictiveRiskSignal:
    reliability = historical_intelligence.precision_for(descriptor.analyzer_id)
    confidence = compose_forecast_confidence(
        evidence_strength=min(1.0, max(0.0, evidence_strength)),
        analyzer_reliability=reliability,
        data_completeness=subject.data_completeness,
    )
    model_version = f"{descriptor.analyzer_id}@{descriptor.version}"
    return PredictiveRiskSignal(
        signal_id=mint_predictive_signal_id(),
        prediction_run_id=PREDICTION_RUN_ID_ENGINE_STAMP,
        tenant_id=tenant_id,
        scope=PredictiveRiskScope.COMPONENT,
        subject_identity=subject.subject_identity,
        risk_type=risk_type,
        severity=severity,
        confidence=confidence,
        evidence_refs=evidence_refs,
        prediction_window=PredictiveWindow(
            duration_seconds=int(timedelta(minutes=window_minutes).total_seconds()),
            label=window_label,
        ),
        generated_at=datetime.now(tz=UTC),
        analyzer_metadata=PredictiveAnalyzerMetadata(
            analyzer_id=descriptor.analyzer_id,
            analyzer_version=model_version,
        ),
        model_version=model_version,
        summary=summary,
        recommended_actions=(),
    )


__all__ = ["emit_forecast_signal"]
