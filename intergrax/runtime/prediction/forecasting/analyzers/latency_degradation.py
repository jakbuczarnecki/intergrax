# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Latency degradation statistical forecast (PREDICTIVE R3)."""

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
_MIN_GROWTH = 0.35

_DESCRIPTOR = ForecastAnalyzerDescriptor(
    analyzer_id="latency_degradation_forecast",
    version="1.0.0",
    namespace=_PLATFORM_NS,
    priority=110,
    supported_features=("latency_growth_rate",),
    resource_budget=ForecastResourceBudget(
        max_execution_time_ms=50,
        max_input_points=1000,
        max_memory_kb=256,
    ),
    supported_risk_types=("LATENCY_DEGRADATION",),
)


@dataclass(frozen=True, slots=True)
class LatencyDegradationForecastAnalyzer:
    """Detects sustained latency slope — not root cause."""

    def analyze(
        self,
        features: PredictiveFeatureSet,
        *,
        historical_intelligence: HistoricalRiskIntelligence,
    ) -> tuple[PredictiveRiskSignal, ...]:
        signals: list[PredictiveRiskSignal] = []
        for subject in features.subjects:
            growth = subject.latency_growth_rate
            if growth is None or growth < _MIN_GROWTH:
                continue
            evidence_strength = min(0.98, 0.55 + growth * 0.4)
            severity = PredictiveRiskSeverity.HIGH if growth >= 0.6 else PredictiveRiskSeverity.MEDIUM
            signals.append(
                emit_forecast_signal(
                    tenant_id=features.tenant_id,
                    subject=subject,
                    descriptor=_DESCRIPTOR,
                    risk_type="LATENCY_DEGRADATION",
                    severity=severity,
                    evidence_strength=evidence_strength,
                    historical_intelligence=historical_intelligence,
                    evidence_refs=(
                        f"feature:latency_growth_rate:{growth:.3f}",
                        f"subject:{subject.subject_identity}",
                    ),
                    summary=(
                        f"{subject.subject_identity} latency trend indicates degradation "
                        f"(normalized slope {growth:.2f})."
                    ),
                    window_minutes=30,
                    window_label="20-40 min",
                ),
            )
        return tuple(signals)

    @property
    def descriptor(self) -> ForecastAnalyzerDescriptor:
        return _DESCRIPTOR


__all__ = ["LatencyDegradationForecastAnalyzer"]
