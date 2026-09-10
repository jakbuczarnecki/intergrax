# © Artur Czarnecki. All rights reserved.

"""Deterministic Markdown projection for R3 qualification performance evidence."""

from __future__ import annotations

from testing_support.execution_qualification.performance_snapshot import (
    QualificationPerformanceSnapshot,
    QualificationSuiteTimingRow,
    build_suite_timing_rows_by_duration_desc,
)


def format_suite_timing_table(rows: tuple[QualificationSuiteTimingRow, ...]) -> str:
    lines = [
        "| Frozen label | Duration (s) | Status | Outcome | Relative child share |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in rows:
        share = (
            f"{row.child_duration_share * 100:.1f}%"
            if row.child_duration_share is not None
            else "n/a"
        )
        lines.append(
            f"| {row.display_label} | {row.duration_seconds:.2f} | {row.status.value} | "
            f"{row.outcome_kind.value} | {share} |"
        )
    return "\n".join(lines)


def format_performance_summary(snapshot: QualificationPerformanceSnapshot) -> str:
    return "\n".join(
        [
            f"run_id={snapshot.run_id}",
            f"status={snapshot.run_status.value}",
            f"max_parallel={snapshot.max_parallel}",
            f"suite_count={snapshot.suite_count}",
            f"wall_duration_seconds={snapshot.wall_duration_seconds:.2f}",
            f"sum_child_duration_seconds={snapshot.sum_child_duration_seconds:.2f}",
            f"max_child_duration_seconds={snapshot.max_child_duration_seconds:.2f}",
            f"observed_overlap_ratio={snapshot.observed_overlap_ratio:.3f}",
            f"artifact_root={snapshot.artifact_root!s}",
        ]
    )


def format_bottleneck_view(manifest_order_rows: tuple[QualificationSuiteTimingRow, ...]) -> str:
    ranked = build_suite_timing_rows_by_duration_desc(manifest_order_rows)
    return format_suite_timing_table(ranked)
