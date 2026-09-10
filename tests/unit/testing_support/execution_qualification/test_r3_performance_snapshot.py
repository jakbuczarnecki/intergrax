# © Artur Czarnecki. All rights reserved.

"""R3 performance snapshot contracts and overlap metric proofs."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import (
    QualificationCoordinator,
    validate_and_run_measured,
)
from testing_support.execution_qualification.performance_snapshot import (
    QualificationPerformanceSnapshot,
    build_performance_snapshot,
    build_suite_timing_rows,
    compute_observed_overlap_ratio,
)
from testing_support.execution_qualification.frozen_pytest_adapter import adapt_frozen_pytest_suites
from tests.unit.runtime.architecture.npsc5e_r3_final_execution_qualification import (
    NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL,
    NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID,
)
from tests.unit.testing_support.execution_qualification.fake_executor import (
    FakeQualificationSuiteExecutor,
)


def test_compute_observed_overlap_ratio_serial_equivalent() -> None:
    assert compute_observed_overlap_ratio(100.0, 100.0) == 1.0


def test_compute_observed_overlap_ratio_both_zero() -> None:
    assert compute_observed_overlap_ratio(0.0, 0.0) == 0.0


def test_compute_observed_overlap_ratio_rejects_non_positive_wall_with_child_work() -> None:
    with pytest.raises(ValueError, match="wall_duration_seconds must be positive"):
        compute_observed_overlap_ratio(10.0, 0.0)
    with pytest.raises(ValueError, match="wall_duration_seconds must be positive"):
        compute_observed_overlap_ratio(10.0, -1.0)


def test_compute_observed_overlap_ratio_with_parallel_child_work() -> None:
    ratio = compute_observed_overlap_ratio(200.0, 100.0)
    assert ratio == 2.0


def test_build_performance_snapshot_fields() -> None:
    suite = ExecutionQualificationSuiteResult(
        suite_id="a",
        command=("uv", "run", "pytest"),
        status=QualificationSuiteStatus.PASS,
        outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
        exit_code=0,
        duration_seconds=30.0,
        log_path=Path("build/qualification/run/a.log"),
    )
    other = ExecutionQualificationSuiteResult(
        suite_id="b",
        command=("uv", "run", "pytest"),
        status=QualificationSuiteStatus.PASS,
        outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
        exit_code=0,
        duration_seconds=70.0,
        log_path=Path("build/qualification/run/b.log"),
    )
    result = ExecutionQualificationRunResult(
        run_id="run-1",
        status=QualificationRunStatus.PASS,
        suite_results=(suite, other),
    )
    snapshot = build_performance_snapshot(
        result,
        wall_duration_seconds=80.0,
        max_parallel=2,
        artifact_root=Path("build/qualification/run-1"),
    )
    assert snapshot.sum_child_duration_seconds == 100.0
    assert snapshot.max_child_duration_seconds == 70.0
    assert snapshot.observed_overlap_ratio == 1.25
    assert snapshot.wall_duration_seconds == 80.0
    assert snapshot.suite_count == 2
    assert isinstance(snapshot.artifact_root, Path)


def test_suite_timing_rows_manifest_order_and_shares() -> None:
    source = (
        ("Terminal", ["tests/unit/runtime/execution/test_p0c6_terminal_outcome_convergence.py"]),
        ("Fan-out", ["tests/unit/agent_distribution/test_bounded_multi_agent_fanout.py"]),
    )
    adapted = adapt_frozen_pytest_suites(
        source,
        label_to_suite_id={
            "Terminal": NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID["Terminal"],
            "Fan-out": NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID["Fan-out"],
        },
    )
    terminal_id = adapted[0].suite.suite_id
    fanout_id = adapted[1].suite.suite_id
    result = ExecutionQualificationRunResult(
        run_id="r3-timing-rows",
        status=QualificationRunStatus.PASS,
        suite_results=(
            ExecutionQualificationSuiteResult(
                suite_id=terminal_id,
                command=("uv", "run", "pytest"),
                status=QualificationSuiteStatus.PASS,
                outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
                exit_code=0,
                duration_seconds=10.0,
                log_path=Path("build/qualification/x") / f"{terminal_id}.log",
            ),
            ExecutionQualificationSuiteResult(
                suite_id=fanout_id,
                command=("uv", "run", "pytest"),
                status=QualificationSuiteStatus.PASS,
                outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
                exit_code=0,
                duration_seconds=30.0,
                log_path=Path("build/qualification/x") / f"{fanout_id}.log",
            ),
        ),
    )
    rows = build_suite_timing_rows(adapted, result)
    assert tuple(row.display_label for row in rows) == ("Terminal", "Fan-out")
    assert rows[0].duration_seconds == 10.0
    assert rows[1].duration_seconds == 30.0
    assert rows[0].child_duration_share == pytest.approx(0.25)
    assert rows[1].child_duration_share == pytest.approx(0.75)
    assert rows[0].status is QualificationSuiteStatus.PASS
    assert rows[0].outcome_kind is QualificationSuiteOutcomeKind.COMPLETED
    assert isinstance(rows[0].log_path, Path)


def _minimal_snapshot(**overrides: float) -> QualificationPerformanceSnapshot:
    base = dict(
        run_id="run",
        run_status=QualificationRunStatus.PASS,
        max_parallel=2,
        suite_count=1,
        wall_duration_seconds=10.0,
        sum_child_duration_seconds=10.0,
        max_child_duration_seconds=10.0,
        observed_overlap_ratio=1.0,
        artifact_root=Path("build/run"),
    )
    base.update(overrides)
    return QualificationPerformanceSnapshot(**base)


def test_performance_snapshot_rejects_non_finite_wall() -> None:
    with pytest.raises(ValueError, match="wall_duration_seconds must be finite"):
        _minimal_snapshot(wall_duration_seconds=float("nan"))
    with pytest.raises(ValueError, match="wall_duration_seconds must be finite"):
        _minimal_snapshot(wall_duration_seconds=float("inf"))


def test_performance_snapshot_rejects_non_finite_child_sum() -> None:
    with pytest.raises(ValueError, match="sum_child_duration_seconds must be finite"):
        _minimal_snapshot(sum_child_duration_seconds=float("nan"))


def test_performance_snapshot_rejects_non_finite_overlap() -> None:
    with pytest.raises(ValueError, match="observed_overlap_ratio must be finite"):
        _minimal_snapshot(observed_overlap_ratio=float("inf"))


def test_qualification_coordinator_run_returns_execution_qualification_run_result(
    repo_root: Path,
) -> None:
    manifest = QualificationRunManifest(
        suites=(QualificationSuite(suite_id="only", pytest_arguments=("x",)),),
    )
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=1,
        run_artifact_root=repo_root / "build" / "qualification" / "r3-run-compat",
        suite_timeout_seconds=60.0,
        run_id="r3-run-compat",
    )
    result = QualificationCoordinator(
        executor=FakeQualificationSuiteExecutor({}),
    ).run(manifest, config)
    assert isinstance(result, ExecutionQualificationRunResult)


def test_max_parallel_two_demonstrates_overlap_with_fake_parallel_work(
    repo_root: Path,
) -> None:
    suites = (
        QualificationSuite(suite_id="slow-a", pytest_arguments=("a",)),
        QualificationSuite(suite_id="slow-b", pytest_arguments=("b",)),
        QualificationSuite(suite_id="slow-c", pytest_arguments=("c",)),
        QualificationSuite(suite_id="slow-d", pytest_arguments=("d",)),
    )
    manifest = QualificationRunManifest(suites=suites)
    config = QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=2,
        run_artifact_root=repo_root / "build" / "qualification" / "r3-overlap-proof",
        suite_timeout_seconds=60.0,
        run_id="r3-overlap-proof",
    )
    measured = validate_and_run_measured(
        manifest,
        config,
        executor=FakeQualificationSuiteExecutor({}, default_sleep_seconds=0.15),
    )
    assert measured.performance.max_parallel == 2
    assert measured.performance.observed_overlap_ratio > 1.0
    assert NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL == 2


def test_qualified_max_parallel_policy_remains_two() -> None:
    assert NPSC5E_R3_EXECUTION_QUALIFICATION_MAX_PARALLEL == 2
