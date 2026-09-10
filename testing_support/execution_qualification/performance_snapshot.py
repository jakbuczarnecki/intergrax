# © Artur Czarnecki. All rights reserved.

"""Typed performance evidence projection for execution qualification runs (R3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationRunStatus,
)
from testing_support.execution_qualification.frozen_pytest_adapter import AdaptedMandatorySuite


@dataclass(frozen=True, slots=True)
class QualificationSuiteTimingRow:
    suite_id: str
    display_label: str
    status: str
    outcome_kind: str
    duration_seconds: float
    log_path: str
    child_duration_share: float | None


@dataclass(frozen=True, slots=True)
class QualificationPerformanceSnapshot:
    """Observed run-level timing; wall clock is measured independently of child sums."""

    run_id: str
    run_status: QualificationRunStatus
    max_parallel: int
    suite_count: int
    wall_duration_seconds: float
    sum_child_duration_seconds: float
    max_child_duration_seconds: float
    observed_overlap_ratio: float
    artifact_root: str

    def __post_init__(self) -> None:
        if self.wall_duration_seconds < 0:
            raise ValueError("wall_duration_seconds must be non-negative")
        if self.sum_child_duration_seconds < 0:
            raise ValueError("sum_child_duration_seconds must be non-negative")
        if self.max_child_duration_seconds < 0:
            raise ValueError("max_child_duration_seconds must be non-negative")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if self.suite_count < 1:
            raise ValueError("suite_count must be >= 1")
        if self.observed_overlap_ratio < 0:
            raise ValueError("observed_overlap_ratio must be non-negative")


@dataclass(frozen=True, slots=True)
class ExecutionQualificationMeasuredRun:
    result: ExecutionQualificationRunResult
    performance: QualificationPerformanceSnapshot


def _suite_result_by_id(
    suite_results: Sequence[ExecutionQualificationSuiteResult],
) -> dict[str, ExecutionQualificationSuiteResult]:
    return {row.suite_id: row for row in suite_results}


def build_suite_timing_rows(
    adapted: Sequence[AdaptedMandatorySuite],
    result: ExecutionQualificationRunResult,
    *,
    label_by_suite_id: Mapping[str, str] | None = None,
) -> tuple[QualificationSuiteTimingRow, ...]:
    """Manifest declaration order; includes failed/skipped suites."""
    by_id = _suite_result_by_id(result.suite_results)
    sum_child = sum(row.duration_seconds for row in result.suite_results)
    rows: list[QualificationSuiteTimingRow] = []
    for entry in adapted:
        suite_id = entry.suite.suite_id
        suite_result = by_id.get(suite_id)
        if suite_result is None:
            raise ValueError(f"missing suite result for suite_id={suite_id!r}")
        label = (
            label_by_suite_id.get(suite_id, entry.display_label)
            if label_by_suite_id is not None
            else entry.display_label
        )
        share: float | None = None
        if sum_child > 0:
            share = suite_result.duration_seconds / sum_child
        rows.append(
            QualificationSuiteTimingRow(
                suite_id=suite_id,
                display_label=label,
                status=suite_result.status.value,
                outcome_kind=suite_result.outcome_kind.value,
                duration_seconds=suite_result.duration_seconds,
                log_path=str(suite_result.log_path),
                child_duration_share=share,
            )
        )
    return tuple(rows)


def build_suite_timing_rows_by_duration_desc(
    manifest_order_rows: tuple[QualificationSuiteTimingRow, ...],
) -> tuple[QualificationSuiteTimingRow, ...]:
    return tuple(sorted(manifest_order_rows, key=lambda row: (-row.duration_seconds, row.suite_id)))


def compute_observed_overlap_ratio(
    sum_child_duration_seconds: float,
    wall_duration_seconds: float,
) -> float:
    if wall_duration_seconds <= 0:
        return 0.0 if sum_child_duration_seconds <= 0 else float("inf")
    return sum_child_duration_seconds / wall_duration_seconds


def build_performance_snapshot(
    result: ExecutionQualificationRunResult,
    *,
    wall_duration_seconds: float,
    max_parallel: int,
    artifact_root: str,
) -> QualificationPerformanceSnapshot:
    durations = tuple(row.duration_seconds for row in result.suite_results)
    sum_child = sum(durations)
    max_child = max(durations) if durations else 0.0
    overlap = compute_observed_overlap_ratio(sum_child, wall_duration_seconds)
    return QualificationPerformanceSnapshot(
        run_id=result.run_id,
        run_status=result.status,
        max_parallel=max_parallel,
        suite_count=len(result.suite_results),
        wall_duration_seconds=wall_duration_seconds,
        sum_child_duration_seconds=sum_child,
        max_child_duration_seconds=max_child,
        observed_overlap_ratio=overlap,
        artifact_root=artifact_root,
    )


def attach_performance_snapshot(
    result: ExecutionQualificationRunResult,
    *,
    wall_duration_seconds: float,
    max_parallel: int,
    artifact_root: str,
) -> ExecutionQualificationMeasuredRun:
    performance = build_performance_snapshot(
        result,
        wall_duration_seconds=wall_duration_seconds,
        max_parallel=max_parallel,
        artifact_root=artifact_root,
    )
    return ExecutionQualificationMeasuredRun(result=result, performance=performance)
