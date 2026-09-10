# © Artur Czarnecki. All rights reserved.

from pathlib import Path

from testing_support.execution_qualification.performance_evidence import (
    format_performance_summary,
    format_suite_timing_table,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunStatus,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.performance_snapshot import (
    QualificationPerformanceSnapshot,
    QualificationSuiteTimingRow,
)


def test_format_suite_timing_table_deterministic() -> None:
    rows = (
        QualificationSuiteTimingRow(
            suite_id="a",
            display_label="A",
            status=QualificationSuiteStatus.PASS,
            outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
            duration_seconds=1.5,
            log_path=Path("build/a.log"),
            child_duration_share=0.5,
        ),
        QualificationSuiteTimingRow(
            suite_id="b",
            display_label="B",
            status=QualificationSuiteStatus.PASS,
            outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
            duration_seconds=1.5,
            log_path=Path("build/b.log"),
            child_duration_share=0.5,
        ),
    )
    table = format_suite_timing_table(rows)
    assert "| A | 1.50 | PASS |" in table
    assert "| B | 1.50 | PASS |" in table


def test_format_performance_summary_includes_overlap() -> None:
    snapshot = QualificationPerformanceSnapshot(
        run_id="run",
        run_status=QualificationRunStatus.PASS,
        max_parallel=2,
        suite_count=2,
        wall_duration_seconds=100.0,
        sum_child_duration_seconds=150.0,
        max_child_duration_seconds=90.0,
        observed_overlap_ratio=1.5,
        artifact_root=Path("build/qualification/run"),
    )
    text = format_performance_summary(snapshot)
    assert "observed_overlap_ratio=1.500" in text
    assert "max_parallel=2" in text
