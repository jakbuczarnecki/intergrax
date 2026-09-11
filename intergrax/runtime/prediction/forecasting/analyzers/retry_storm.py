# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Retry amplification statistical forecast (PREDICTIVE R3)."""

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
_BASELINE_RETRY = 1.2
_STORM_RATIO = 3.0

_DESCRIPTOR = ForecastAnalyzerDescriptor(
    analyzer_id="retry_storm_forecast",
    version="1.0.0",
    namespace=_PLATFORM_NS,
    priority=100,
    supported_features=("retry_per_execution", "retry_growth_rate"),
    resource_budget=ForecastResourceBudget(
        max_execution_time_ms=50,
        max_input_points=1000,
        max_memory_kb=256,
    ),
    supported_risk_types=("RETRY_STORM_RISK",),
)


@dataclass(frozen=True, slots=True)
class RetryStormForecastAnalyzer:
    """Detects retry amplification vs baseline — not incident classification."""

    def analyze(
        self,
        features: PredictiveFeatureSet,
        *,
        historical_intelligence: HistoricalRiskIntelligence,
    ) -> tuple[PredictiveRiskSignal, ...]:
        signals: list[PredictiveRiskSignal] = []
        for subject in features.subjects:
            current = subject.retry_per_execution
            if current is None:
                continue
            ratio = current / _BASELINE_RETRY
            if ratio < _STORM_RATIO:
                continue
            growth = subject.retry_growth_rate or 0.0
            evidence_strength = min(0.96, 0.45 + (ratio - _STORM_RATIO) * 0.15 + growth * 0.25)
            severity = PredictiveRiskSeverity.HIGH if ratio >= 4.0 else PredictiveRiskSeverity.MEDIUM
            signals.append(
                emit_forecast_signal(
                    tenant_id=features.tenant_id,
                    subject=subject,
                    descriptor=_DESCRIPTOR,
                    risk_type="RETRY_STORM_RISK",
                    severity=severity,
                    evidence_strength=evidence_strength,
                    historical_intelligence=historical_intelligence,
                    evidence_refs=(
                        f"feature:retry_per_execution:{current:.2f}",
                        f"baseline_retry:{_BASELINE_RETRY}",
                        f"subject:{subject.subject_identity}",
                    ),
                    summary=(
                        f"{subject.subject_identity} retry amplification "
                        f"({current:.1f} retries/execution vs {_BASELINE_RETRY} baseline)."
                    ),
                    window_minutes=45,
                    window_label="next 45 min",
                ),
            )
        return tuple(signals)

    @property
    def descriptor(self) -> ForecastAnalyzerDescriptor:
        return _DESCRIPTOR


__all__ = ["RetryStormForecastAnalyzer"]
