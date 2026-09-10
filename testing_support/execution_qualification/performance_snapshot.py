# © Artur Czarnecki. All rights reserved.

"""Typed performance evidence projection for execution qualification runs (R3)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationRunStatus,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.frozen_pytest_adapter import AdaptedMandatorySuite


def _require_finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class QualificationSuiteTimingRow:
    suite_id: str
    display_label: str
    status: QualificationSuiteStatus
    outcome_kind: QualificationSuiteOutcomeKind
    duration_seconds: float
    log_path: Path
    child_duration_share: float | None

    def __post_init__(self) -> None:
        _require_finite_non_negative("duration_seconds", self.duration_seconds)
        if self.child_duration_share is not None:
            if not math.isfinite(self.child_duration_share):
                raise ValueError("child_duration_share must be finite")
            if self.child_duration_share < 0:
                raise ValueError("child_duration_share must be non-negative")


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
    artifact_root: Path

    def __post_init__(self) -> None:
        _require_finite_non_negative("wall_duration_seconds", self.wall_duration_seconds)
        _require_finite_non_negative(
            "sum_child_duration_seconds",
            self.sum_child_duration_seconds,
        )
        _require_finite_non_negative(
            "max_child_duration_seconds",
            self.max_child_duration_seconds,
        )
        _require_finite_non_negative(
            "observed_overlap_ratio",
            self.observed_overlap_ratio,
        )
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if self.suite_count < 1:
            raise ValueError("suite_count must be >= 1")
        if self.max_child_duration_seconds > self.sum_child_duration_seconds:
            raise ValueError(
                "max_child_duration_seconds must not exceed sum_child_duration_seconds"
            )


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
                status=suite_result.status,
                outcome_kind=suite_result.outcome_kind,
                duration_seconds=suite_result.duration_seconds,
                log_path=suite_result.log_path,
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
    if sum_child_duration_seconds == 0 and wall_duration_seconds == 0:
        return 0.0
    if wall_duration_seconds <= 0:
        raise ValueError(
            "wall_duration_seconds must be positive when child work duration is non-zero"
        )
    return sum_child_duration_seconds / wall_duration_seconds


def build_performance_snapshot(
    result: ExecutionQualificationRunResult,
    *,
    wall_duration_seconds: float,
    max_parallel: int,
    artifact_root: Path,
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
    artifact_root: Path,
) -> ExecutionQualificationMeasuredRun:
    performance = build_performance_snapshot(
        result,
        wall_duration_seconds=wall_duration_seconds,
        max_parallel=max_parallel,
        artifact_root=artifact_root,
    )
    return ExecutionQualificationMeasuredRun(result=result, performance=performance)
