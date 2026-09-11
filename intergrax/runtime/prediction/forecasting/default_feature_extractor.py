# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default bounded feature extraction from PredictiveContext (PREDICTIVE R3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.predictive_context import PerformanceMetricPoint, PredictiveContext
from intergrax.contracts.predictive_feature_set import PredictiveFeatureSet, SubjectPredictiveFeatures

MAX_POINTS_PER_METRIC = 1000
_MINUTES_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class DefaultPredictiveFeatureExtractor:
    """Derives slopes and rates from performance and execution pattern snapshots."""

    max_points_per_metric: int = MAX_POINTS_PER_METRIC

    def extract(self, context: PredictiveContext) -> PredictiveFeatureSet:
        subjects = _collect_subject_ids(context)
        subject_features: list[SubjectPredictiveFeatures] = []
        completeness_scores: list[float] = []

        for subject in subjects:
            pattern = _pattern_for(context, subject)
            latency_points = _metric_series(
                context.performance_history,
                metric_name="latency_ms",
                component_id=subject,
                limit=self.max_points_per_metric,
            )
            retry_points = _metric_series(
                context.performance_history,
                metric_name="retry_per_execution",
                component_id=subject,
                limit=self.max_points_per_metric,
            )
            memory_points = _metric_series(
                context.performance_history,
                metric_name="memory_utilization_pct",
                component_id=subject,
                limit=self.max_points_per_metric,
            )
            failure_points = _metric_series(
                context.failure_history,
                metric_name="failure_count",
                component_id=subject,
                limit=self.max_points_per_metric,
            )

            latency_slope = _latency_growth_rate(latency_points)
            retry_growth = _normalized_slope(retry_points)
            memory_slope = _normalized_slope(memory_points)
            failure_accel = _failure_frequency_delta(failure_points)

            failure_rate: float | None = None
            execution_variance: float | None = None
            retry_current: float | None = None
            if pattern is not None and pattern.execution_count > 0:
                failure_rate = pattern.failed_execution_count / pattern.execution_count
                if pattern.avg_latency_ms is not None:
                    execution_variance = pattern.avg_latency_ms / max(pattern.execution_count, 1)
            if retry_points:
                retry_current = retry_points[-1].value

            memory_latest = memory_points[-1].value if memory_points else None

            completeness_parts: list[float] = []
            if latency_points:
                completeness_parts.append(min(1.0, len(latency_points) / 3.0))
            if failure_points:
                completeness_parts.append(min(1.0, len(failure_points) / 3.0))
            if retry_points:
                completeness_parts.append(min(1.0, len(retry_points) / 2.0))
            if memory_points:
                completeness_parts.append(min(1.0, len(memory_points) / 3.0))
            if pattern is not None and pattern.execution_count > 0:
                completeness_parts.append(min(1.0, pattern.execution_count / 10.0))
            completeness = max(completeness_parts) if completeness_parts else 0.0

            subject_features.append(
                SubjectPredictiveFeatures(
                    subject_identity=subject,
                    latency_growth_rate=latency_slope,
                    failure_rate=failure_rate,
                    failure_frequency_delta=failure_accel,
                    retry_per_execution=retry_current,
                    retry_growth_rate=retry_growth,
                    execution_variance=execution_variance,
                    resource_utilization_latest=memory_latest,
                    resource_utilization_slope=memory_slope,
                    data_completeness=completeness,
                ),
            )
            completeness_scores.append(completeness)

        global_completeness = (
            sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        )
        return PredictiveFeatureSet(
            tenant_id=context.tenant_id,
            input_snapshot_id=context.input_snapshot_id,
            as_of=context.as_of,
            subjects=tuple(subject_features),
            global_data_completeness=global_completeness,
        )


def _collect_subject_ids(context: PredictiveContext) -> tuple[str, ...]:
    ids: set[str] = {p.subject_identity for p in context.execution_patterns}
    for point in context.performance_history:
        if point.component_id:
            ids.add(point.component_id)
    for point in context.failure_history:
        if point.component_id:
            ids.add(point.component_id)
    return tuple(sorted(ids))


def _pattern_for(context: PredictiveContext, subject: str):
    for pattern in context.execution_patterns:
        if pattern.subject_identity == subject:
            return pattern
    return None


def _metric_series(
    points: tuple[PerformanceMetricPoint, ...],
    *,
    metric_name: str,
    component_id: str,
    limit: int,
) -> tuple[PerformanceMetricPoint, ...]:
    matched = [
        p
        for p in points
        if p.metric_name == metric_name
        and (p.component_id is None or p.component_id == component_id)
    ]
    ordered = sorted(matched, key=lambda p: p.observed_at)
    if len(ordered) > limit:
        ordered = ordered[-limit:]
    return tuple(ordered)


def _latency_growth_rate(points: tuple[PerformanceMetricPoint, ...]) -> float | None:
    if len(points) < 2:
        return None
    ordered = sorted(points, key=lambda p: p.observed_at)
    baseline = ordered[0].value
    latest = ordered[-1].value
    if baseline <= 0:
        return None
    relative = (latest - baseline) / baseline
    if relative <= 0:
        return None
    slope_norm = _normalized_slope(ordered)
    if slope_norm is None:
        return relative
    return max(relative, slope_norm)


def _normalized_slope(points: tuple[PerformanceMetricPoint, ...]) -> float | None:
    if len(points) < 2:
        return None
    t0 = points[0].observed_at
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        minutes = (point.observed_at - t0).total_seconds() / 60.0
        xs.append(minutes)
        ys.append(point.value)
    if max(xs) < _MINUTES_EPS:
        return None
    slope = _least_squares_slope(xs, ys)
    baseline = ys[0]
    if baseline <= 0:
        return slope
    return slope / baseline


def _least_squares_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den <= 0:
        return 0.0
    return num / den


def _failure_frequency_delta(points: tuple[PerformanceMetricPoint, ...]) -> float | None:
    if len(points) < 2:
        return None
    first = points[0].value
    last = points[-1].value
    if first <= 0:
        return None if last <= first else last - first
    return (last - first) / first


__all__ = ["DefaultPredictiveFeatureExtractor", "MAX_POINTS_PER_METRIC"]
