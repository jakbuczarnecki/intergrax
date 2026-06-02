# © Artur Czarnecki. All rights reserved.

"""Architecture metrics emission and trend guard contracts (Phase V-AM.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.architecture_metrics import ArchitectureMetricsReport


class MetricsTrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


class ArchitectureMetricsSnapshot(BaseModel):
    snapshot_id: str
    collected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    report: ArchitectureMetricsReport


class ArchitectureMetricsTrend(BaseModel):
    modularity_trend: MetricsTrendDirection
    dependency_health_trend: MetricsTrendDirection
    observability_coverage_trend: MetricsTrendDirection
    governance_coverage_trend: MetricsTrendDirection
    architecture_debt_trend: MetricsTrendDirection


class ArchitectureMetricsGateResult(BaseModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class ArchitectureMetricsPipelineReport(BaseModel):
    schema_version: str = "1.0.0"
    mode: str = "report-and-enforce"
    snapshots: list[ArchitectureMetricsSnapshot] = Field(default_factory=list)
    trend: ArchitectureMetricsTrend | None = None
    gate_result: ArchitectureMetricsGateResult


def build_metrics_pipeline_report(
    *,
    snapshots: list[ArchitectureMetricsSnapshot],
) -> ArchitectureMetricsPipelineReport:
    gate_result = _evaluate_gate(snapshots)
    trend = _compute_trend(snapshots) if len(snapshots) >= 2 else None
    return ArchitectureMetricsPipelineReport(
        snapshots=snapshots,
        trend=trend,
        gate_result=gate_result,
    )


def _evaluate_gate(snapshots: list[ArchitectureMetricsSnapshot]) -> ArchitectureMetricsGateResult:
    if not snapshots:
        return ArchitectureMetricsGateResult(passed=False, reasons=["No architecture metric snapshots"])

    latest = snapshots[-1].report
    thresholds = latest.thresholds
    summary = latest.summary
    reasons: list[str] = []

    if summary.modularity_score < thresholds.modularity_score_min:
        reasons.append("Modularity score below threshold")
    if summary.dependency_health_score < thresholds.dependency_health_score_min:
        reasons.append("Dependency health score below threshold")
    if summary.observability_coverage < thresholds.observability_coverage_min:
        reasons.append("Observability coverage below threshold")
    if summary.governance_coverage < thresholds.governance_coverage_min:
        reasons.append("Governance coverage below threshold")
    if summary.architecture_debt_index > thresholds.architecture_debt_index_max:
        reasons.append("Architecture debt index above threshold")

    return ArchitectureMetricsGateResult(passed=not reasons, reasons=reasons)


def _compute_trend(snapshots: list[ArchitectureMetricsSnapshot]) -> ArchitectureMetricsTrend:
    previous = snapshots[-2].report.summary
    current = snapshots[-1].report.summary
    return ArchitectureMetricsTrend(
        modularity_trend=_positive_metric_trend(previous.modularity_score, current.modularity_score),
        dependency_health_trend=_positive_metric_trend(
            previous.dependency_health_score,
            current.dependency_health_score,
        ),
        observability_coverage_trend=_positive_metric_trend(
            previous.observability_coverage,
            current.observability_coverage,
        ),
        governance_coverage_trend=_positive_metric_trend(
            previous.governance_coverage,
            current.governance_coverage,
        ),
        architecture_debt_trend=_negative_metric_trend(
            previous.architecture_debt_index,
            current.architecture_debt_index,
        ),
    )


def _positive_metric_trend(previous: float, current: float) -> MetricsTrendDirection:
    if current > previous:
        return MetricsTrendDirection.IMPROVING
    if current < previous:
        return MetricsTrendDirection.DEGRADING
    return MetricsTrendDirection.STABLE


def _negative_metric_trend(previous: float, current: float) -> MetricsTrendDirection:
    if current < previous:
        return MetricsTrendDirection.IMPROVING
    if current > previous:
        return MetricsTrendDirection.DEGRADING
    return MetricsTrendDirection.STABLE
