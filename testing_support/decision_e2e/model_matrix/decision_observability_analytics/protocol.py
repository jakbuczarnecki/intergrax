# © Artur Czarnecki. All rights reserved.

"""Pluggable observability contracts (DS-E2E-15J-L8)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.decision_observability_analytics.contracts import (
    DecisionAnalyticsReport,
    DecisionAnalyticsResult,
    DecisionMetricsSnapshot,
    DecisionObservation,
)


class DecisionObservationCollector(Protocol):
    """Collect factual decision observations from a data source."""

    @property
    def collector_id(self) -> str: ...

    def collect(self) -> tuple[DecisionObservation, ...]: ...


class DecisionAnalyticsAnalyzer(Protocol):
    """Pluggable analytics over collected observations."""

    @property
    def analyzer_id(self) -> str: ...

    @property
    def analyzer_version(self) -> str: ...

    def analyze(
        self,
        observations: tuple[DecisionObservation, ...],
        *,
        analyzed_at: datetime,
    ) -> DecisionAnalyticsResult: ...


class DecisionMetricsProvider(Protocol):
    """Pluggable metrics derived from observations."""

    @property
    def metrics_provider_id(self) -> str: ...

    @property
    def metrics_provider_version(self) -> str: ...

    def collect_metrics(
        self,
        observations: tuple[DecisionObservation, ...],
    ) -> DecisionMetricsSnapshot: ...


class DecisionAnalyticsReportProvider(Protocol):
    """Pluggable report assembly from analytics results."""

    @property
    def report_provider_id(self) -> str: ...

    @property
    def report_provider_version(self) -> str: ...

    def build_report(
        self,
        results: tuple[DecisionAnalyticsResult, ...],
        *,
        generated_at: datetime,
    ) -> DecisionAnalyticsReport: ...


__all__ = [
    "DecisionAnalyticsAnalyzer",
    "DecisionAnalyticsReportProvider",
    "DecisionMetricsProvider",
    "DecisionObservationCollector",
]
