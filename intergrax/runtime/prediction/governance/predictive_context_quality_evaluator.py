# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Context completeness, freshness, coverage, reliability (PREDICTIVE R4)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from intergrax.contracts.predictive import PredictiveContext, PredictiveContextCompleteness
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport

_SECTION_LATENCY = "latency_history"
_SECTION_FAILURE = "failure_history"
_SECTION_BUSINESS = "business_metrics"
_SECTION_DIAGNOSTIC = "diagnostic_history"

_RELIABILITY_BY_COMPLETENESS: dict[PredictiveContextCompleteness, float] = {
    PredictiveContextCompleteness.COMPLETE: 1.0,
    PredictiveContextCompleteness.PARTIAL: 0.72,
    PredictiveContextCompleteness.LIMITED: 0.5,
    PredictiveContextCompleteness.UNAVAILABLE: 0.0,
}


@dataclass(frozen=True, slots=True)
class PredictiveContextQualityEvaluator:
    """Scores readonly predictive context — no storage access."""

    freshness_horizon: timedelta = timedelta(hours=24)

    def evaluate(self, context: PredictiveContext) -> PredictiveContextQualityReport:
        present: list[str] = []
        missing: list[str] = []

        if context.performance.latency_series or context.history.latency_patterns:
            present.append(_SECTION_LATENCY)
        else:
            missing.append(_SECTION_LATENCY)

        if context.history.failure_patterns:
            present.append(_SECTION_FAILURE)
        else:
            missing.append(_SECTION_FAILURE)

        if context.performance.throughput_series or context.current_state:
            present.append(_SECTION_BUSINESS)
        else:
            missing.append(_SECTION_BUSINESS)

        if (
            context.diagnostic.historical_problems
            or context.diagnostic.previous_findings
            or context.diagnostic.previous_risk_signals
        ):
            present.append(_SECTION_DIAGNOSTIC)
        else:
            missing.append(_SECTION_DIAGNOSTIC)

        expected = 4
        coverage = len(present) / expected
        freshness = self._freshness_score(context)
        reliability = _RELIABILITY_BY_COMPLETENESS.get(context.completeness, 0.0)
        if missing:
            reliability = min(reliability, coverage)

        return PredictiveContextQualityReport(
            completeness=context.completeness,
            coverage=coverage,
            freshness_score=freshness,
            reliability=reliability,
            missing_sections=tuple(missing),
        )

    def _freshness_score(self, context: PredictiveContext) -> float:
        points = (
            context.performance.latency_series
            + context.history.failure_patterns
            + context.history.retry_patterns
        )
        if not points:
            return 0.5
        latest = max(p.observed_at for p in points)
        age = context.as_of - latest
        if age <= self.freshness_horizon:
            return 1.0
        if age >= self.freshness_horizon * 4:
            return 0.25
        ratio = age / (self.freshness_horizon * 4)
        return max(0.25, 1.0 - ratio)


__all__ = ["PredictiveContextQualityEvaluator"]
