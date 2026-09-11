# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bridge statistical forecast engine into R1 PredictiveAnalyzer registry (PREDICTIVE R3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.predictive_analyzer import PredictiveAnalyzer
from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_risk import PredictiveRiskSignal
from intergrax.runtime.prediction.forecasting.forecast_registry import (
    PredictiveForecastAnalyzerRegistry,
)
from intergrax.runtime.prediction.forecasting.statistical_forecast_engine import (
    StatisticalForecastEngine,
)

_PLATFORM_NS = "intergrax.platform"


@dataclass(slots=True)
class StatisticalForecastPredictiveAnalyzer:
    """
    Single registry entry that runs the full R3 forecast stack.

    Keeps DIAGNOSTIC_AUTHORITY_COUNT = 1 — forecasts remain consumer projections.
    """

    registry: PredictiveForecastAnalyzerRegistry = field(
        default_factory=PredictiveForecastAnalyzerRegistry.platform_default,
    )
    analyzer_id: str = "statistical_forecast_r3"
    analyzer_namespace: str = _PLATFORM_NS
    priority: int = 120
    model_version: str = "statistical_forecast_r3@1.0.0"

    def analyze(self, context: PredictiveContext) -> tuple[PredictiveRiskSignal, ...]:
        engine = StatisticalForecastEngine(registry=self.registry)
        return engine.analyze(context).signals


__all__ = ["StatisticalForecastPredictiveAnalyzer"]
