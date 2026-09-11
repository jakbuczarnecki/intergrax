# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Orchestrates feature extraction and statistical forecast analyzers (PREDICTIVE R3)."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_feature_extractor import PredictiveFeatureExtractor
from intergrax.contracts.predictive_risk import PredictiveRiskSignal, mint_prediction_run_id
from intergrax.runtime.prediction.forecasting.default_feature_extractor import (
    DefaultPredictiveFeatureExtractor,
)
from intergrax.runtime.prediction.forecasting.forecast_registry import (
    PredictiveForecastAnalyzerRegistry,
)
from intergrax.runtime.prediction.prediction_engine import ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE

DEFAULT_FORECAST_TIME_BUDGET_MS = 500


@dataclass(frozen=True, slots=True)
class StatisticalForecastEngineResult:
    signals: tuple[PredictiveRiskSignal, ...]
    analyzer_outcomes: tuple[str, ...]
    degraded: bool


@dataclass(slots=True)
class StatisticalForecastEngine:
    """Consumer of diagnostic context — emits forecast risk signals only."""

    registry: PredictiveForecastAnalyzerRegistry
    feature_extractor: PredictiveFeatureExtractor = DefaultPredictiveFeatureExtractor()
    time_budget_ms: int = DEFAULT_FORECAST_TIME_BUDGET_MS

    def analyze(self, context: PredictiveContext) -> StatisticalForecastEngineResult:
        if context.tenant_id.strip() == "":
            raise ValueError("context.tenant_id required")

        prediction_run_id = mint_prediction_run_id()
        deadline = time.monotonic() + (self.time_budget_ms / 1000.0)
        features = self.feature_extractor.extract(context)
        signals: list[PredictiveRiskSignal] = []
        outcomes: list[str] = []
        degraded = False

        intelligence = context.historical_risk_intelligence

        for analyzer in self.registry.analyzers:
            if time.monotonic() > deadline:
                degraded = True
                outcomes.append(f"{analyzer.descriptor.analyzer_id}:skipped_time_budget")
                break
            try:
                batch = analyzer.analyze(
                    features,
                    historical_intelligence=intelligence,
                )
            except Exception:
                degraded = True
                outcomes.append(
                    f"{analyzer.descriptor.analyzer_id}:{ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE}",
                )
                continue
            for signal in batch:
                if signal.tenant_id != context.tenant_id:
                    raise ValueError("forecast analyzer emitted cross-tenant risk signal")
                signals.append(replace(signal, prediction_run_id=prediction_run_id))
            outcomes.append(f"{analyzer.descriptor.analyzer_id}:ok:{len(batch)}")

        return StatisticalForecastEngineResult(
            signals=tuple(signals),
            analyzer_outcomes=tuple(outcomes),
            degraded=degraded,
        )


__all__ = [
    "DEFAULT_FORECAST_TIME_BUDGET_MS",
    "StatisticalForecastEngine",
    "StatisticalForecastEngineResult",
]
