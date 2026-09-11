# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Resource exhaustion statistical forecast (PREDICTIVE R3)."""

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
_MIN_UTIL = 0.7
_MIN_SLOPE = 0.02

_DESCRIPTOR = ForecastAnalyzerDescriptor(
    analyzer_id="resource_exhaustion_forecast",
    version="1.0.0",
    namespace=_PLATFORM_NS,
    priority=95,
    supported_features=("resource_utilization_latest", "resource_utilization_slope"),
    resource_budget=ForecastResourceBudget(
        max_execution_time_ms=50,
        max_input_points=1000,
        max_memory_kb=256,
    ),
    supported_risk_types=("RESOURCE_EXHAUSTION_RISK",),
)


@dataclass(frozen=True, slots=True)
class ResourceExhaustionForecastAnalyzer:
    """Detects rising utilization toward exhaustion — not proven outage."""

    def analyze(
        self,
        features: PredictiveFeatureSet,
        *,
        historical_intelligence: HistoricalRiskIntelligence,
    ) -> tuple[PredictiveRiskSignal, ...]:
        signals: list[PredictiveRiskSignal] = []
        for subject in features.subjects:
            latest = subject.resource_utilization_latest
            slope = subject.resource_utilization_slope
            if latest is None:
                continue
            if latest < _MIN_UTIL:
                continue
            if slope is None or slope < _MIN_SLOPE:
                if latest < 0.88:
                    continue
            evidence_strength = min(0.95, 0.4 + latest * 0.45 + (slope or 0.0) * 2.0)
            severity = PredictiveRiskSeverity.CRITICAL if latest >= 0.9 else PredictiveRiskSeverity.HIGH
            signals.append(
                emit_forecast_signal(
                    tenant_id=features.tenant_id,
                    subject=subject,
                    descriptor=_DESCRIPTOR,
                    risk_type="RESOURCE_EXHAUSTION_RISK",
                    severity=severity,
                    evidence_strength=evidence_strength,
                    historical_intelligence=historical_intelligence,
                    evidence_refs=(
                        f"feature:resource_utilization:{latest:.2f}",
                        f"feature:resource_slope:{(slope or 0.0):.3f}",
                        f"subject:{subject.subject_identity}",
                    ),
                    summary=(
                        f"{subject.subject_identity} resource utilization trending toward exhaustion "
                        f"({int(latest * 100)}%)."
                    ),
                    window_minutes=120,
                    window_label="next 2h",
                ),
            )
        return tuple(signals)

    @property
    def descriptor(self) -> ForecastAnalyzerDescriptor:
        return _DESCRIPTOR


__all__ = ["ResourceExhaustionForecastAnalyzer"]
