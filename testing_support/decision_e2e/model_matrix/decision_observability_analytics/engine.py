# © Artur Czarnecki. All rights reserved.

"""Decision observability orchestration via injected plugins (DS-E2E-15J-L8)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.analyzers import (
    default_decision_analytics_analyzers,
)
from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    OBSERVABILITY_TASK_ID,
    DecisionObservabilityRunResult,
)
from testing_support.decision_e2e.model_matrix.decision_observability_analytics.metrics import (
    default_metrics_providers,
)
from testing_support.decision_e2e.model_matrix.decision_observability_analytics.protocol import (
    DecisionAnalyticsAnalyzer,
    DecisionAnalyticsReportProvider,
    DecisionMetricsProvider,
    DecisionObservationCollector,
)
from testing_support.decision_e2e.model_matrix.decision_observability_analytics.reporters import (
    default_report_providers,
)


@dataclass(frozen=True, slots=True)
class DecisionObservabilityEngine:
    collectors: tuple[DecisionObservationCollector, ...]
    analyzers: tuple[DecisionAnalyticsAnalyzer, ...]
    metrics_providers: tuple[DecisionMetricsProvider, ...]
    report_providers: tuple[DecisionAnalyticsReportProvider, ...]

    def run(
        self,
        *,
        run_at: datetime | None = None,
    ) -> DecisionObservabilityRunResult:
        stamp = run_at or datetime.now(tz=UTC)
        observations = tuple(
            observation
            for collector in self.collectors
            for observation in collector.collect()
        )
        analytics_results = tuple(
            analyzer.analyze(observations, analyzed_at=stamp)
            for analyzer in self.analyzers
        )
        metrics = tuple(
            provider.collect_metrics(observations)
            for provider in self.metrics_providers
        )
        reports = tuple(
            provider.build_report(analytics_results, generated_at=stamp)
            for provider in self.report_providers
        )
        return DecisionObservabilityRunResult(
            observability_task_id=OBSERVABILITY_TASK_ID,
            run_at=stamp,
            observations=observations,
            analytics_results=analytics_results,
            metrics=metrics,
            reports=reports,
        )


def default_decision_observability_engine(
    collectors: tuple[DecisionObservationCollector, ...],
    *,
    analyzers: tuple[DecisionAnalyticsAnalyzer, ...] | None = None,
    metrics_providers: tuple[DecisionMetricsProvider, ...] | None = None,
    report_providers: tuple[DecisionAnalyticsReportProvider, ...] | None = None,
) -> DecisionObservabilityEngine:
    return DecisionObservabilityEngine(
        collectors=collectors,
        analyzers=analyzers or default_decision_analytics_analyzers(),
        metrics_providers=metrics_providers or default_metrics_providers(),
        report_providers=report_providers or default_report_providers(),
    )


__all__ = [
    "DecisionObservabilityEngine",
    "default_decision_observability_engine",
]
