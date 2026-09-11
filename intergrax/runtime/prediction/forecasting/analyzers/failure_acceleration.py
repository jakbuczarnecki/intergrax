# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Failure rate acceleration statistical forecast (PREDICTIVE R3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.forecast_analyzer_descriptor import (
    ForecastAnalyzerDescriptor,
    ForecastResourceBudget,
)
from intergrax.contracts.predictive_feature_set import PredictiveFeatureSet
from intergrax.contracts.predictive_historical_intelligence import HistoricalRiskIntelligence
from intergrax.contracts.predictive_risk import PredictiveRiskSeverity, PredictiveRiskSignal
from intergrax.runtime.prediction.forecasting.signal_factory import emit_forecast_signal

_PLATFORM_NS = "intergrax.platform"
_MIN_ACCEL = 0.5

_DESCRIPTOR = ForecastAnalyzerDescriptor(
    analyzer_id="failure_acceleration_forecast",
    version="1.0.0",
    namespace=_PLATFORM_NS,
    priority=105,
    supported_features=("failure_frequency_delta", "failure_rate"),
    resource_budget=ForecastResourceBudget(
        max_execution_time_ms=50,
        max_input_points=1000,
        max_memory_kb=256,
    ),
    supported_risk_types=("FAILURE_ACCELERATION_RISK",),
)


@dataclass(frozen=True, slots=True)
class FailureAccelerationForecastAnalyzer:
    """Detects accelerating failure counts — probability language only."""

    def analyze(
        self,
        features: PredictiveFeatureSet,
        *,
        historical_intelligence: HistoricalRiskIntelligence,
    ) -> tuple[PredictiveRiskSignal, ...]:
        signals: list[PredictiveRiskSignal] = []
        for subject in features.subjects:
            accel = subject.failure_frequency_delta
            if accel is None or accel < _MIN_ACCEL:
                continue
            rate = subject.failure_rate or 0.0
            evidence_strength = min(0.97, 0.5 + accel * 0.35 + rate * 0.2)
            severity = PredictiveRiskSeverity.HIGH if accel >= 1.0 else PredictiveRiskSeverity.MEDIUM
            signals.append(
                emit_forecast_signal(
                    tenant_id=features.tenant_id,
                    subject=subject,
                    descriptor=_DESCRIPTOR,
                    risk_type="FAILURE_ACCELERATION_RISK",
                    severity=severity,
                    evidence_strength=evidence_strength,
                    historical_intelligence=historical_intelligence,
                    evidence_refs=(
                        f"feature:failure_frequency_delta:{accel:.3f}",
                        f"feature:failure_rate:{rate:.3f}",
                        f"subject:{subject.subject_identity}",
                    ),
                    summary=(
                        f"{subject.subject_identity} failure acceleration detected "
                        f"(delta {accel:.2f})."
                    ),
                    window_minutes=60,
                    window_label="next hour",
                ),
            )
        return tuple(signals)

    @property
    def descriptor(self) -> ForecastAnalyzerDescriptor:
        return _DESCRIPTOR


__all__ = ["FailureAccelerationForecastAnalyzer"]
