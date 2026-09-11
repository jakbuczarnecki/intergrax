# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Attach related risk signals to operator investigation reads (PREDICTIVE R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticProblemDetail,
    DiagnosticProblemOccurrenceView,
)
from intergrax.runtime.prediction.predictive_context_builder import (
    build_predictive_context_for_investigation,
)
from intergrax.runtime.prediction.predictive_investigation_projection import (
    project_related_risk_signals,
)
from intergrax.contracts.predictive_investigation_read import RelatedPredictiveRiskSignalView
from intergrax.runtime.prediction.prediction_engine import PredictionEngine


@dataclass(slots=True)
class PredictiveInvestigationService:
    """Consumer of diagnostic facts — emits readonly risk projections only."""

    engine: PredictionEngine

    def related_risk_signals(
        self,
        *,
        problem_detail: DiagnosticProblemDetail,
        occurrence: DiagnosticProblemOccurrenceView,
    ) -> tuple[RelatedPredictiveRiskSignalView, ...]:
        if not self.engine.registry.analyzers:
            return ()
        context = build_predictive_context_for_investigation(
            problem_detail=problem_detail,
            occurrence=occurrence,
        )
        result = self.engine.analyze(context)
        return project_related_risk_signals(result.signals)


__all__ = ["PredictiveInvestigationService"]
