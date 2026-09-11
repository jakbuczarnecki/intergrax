# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Attach related risk signals to operator investigation reads (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.predictive_risk import mint_prediction_run_id
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
)
from intergrax.runtime.prediction.predictive_context_builder import (
    build_predictive_context_for_investigation,
)
from intergrax.runtime.prediction.predictive_investigation_projection import (
    project_forecast_risk_signals,
    project_related_risk_signals,
)
from intergrax.runtime.prediction.prediction_engine import PredictionEngine
from intergrax.runtime.prediction.forecasting.statistical_forecast_engine import (
    StatisticalForecastEngine,
)
from intergrax.runtime.prediction.forecasting.forecast_registry import (
    PredictiveForecastAnalyzerRegistry,
)


@dataclass(slots=True)
class PredictiveInvestigationService:
    """Consumer of diagnostic facts — emits readonly risk projections only."""

    engine: PredictionEngine
    forecast_engine: StatisticalForecastEngine | None = None

    def _forecast_engine(self) -> StatisticalForecastEngine:
        if self.forecast_engine is not None:
            return self.forecast_engine
        return StatisticalForecastEngine(
            registry=PredictiveForecastAnalyzerRegistry.platform_default(),
        )

    def related_risk_signals(
        self,
        *,
        problem_detail: DiagnosticProblemDetail,
        occurrence: DiagnosticProblemOccurrenceView,
    ) -> tuple:
        if not self.engine.registry.analyzers:
            return ()
        context = build_predictive_context_for_investigation(
            problem_detail=problem_detail,
            occurrence=occurrence,
        )
        result = self.engine.analyze(context)
        assert self.engine.governance is not None
        context_quality = self.engine.governance.context_evaluator.evaluate(context)
        return project_related_risk_signals(
            result.signals,
            quality_by_signal_id=result.quality_by_signal_id,
            context_quality=context_quality,
        )

    def forecast_risk_signals(
        self,
        *,
        problem_detail: DiagnosticProblemDetail,
        occurrence: DiagnosticProblemOccurrenceView,
    ) -> tuple:
        context = build_predictive_context_for_investigation(
            problem_detail=problem_detail,
            occurrence=occurrence,
        )
        raw = self._forecast_engine().analyze(context)
        assert self.engine.governance is not None
        run_id = (
            raw.signals[0].prediction_run_id
            if raw.signals
            else mint_prediction_run_id()
        )
        governed = self.engine.governance.govern_run(
            context=context,
            prediction_run_id=run_id,
            raw_signals=raw.signals,
            analyzer_outcomes=raw.analyzer_outcomes,
            degraded=raw.degraded,
        )
        context_quality = self.engine.governance.context_evaluator.evaluate(context)
        return project_forecast_risk_signals(
            governed.signals,
            quality_by_signal_id=governed.quality_by_signal_id,
            context_quality=context_quality,
        )


__all__ = ["PredictiveInvestigationService"]
