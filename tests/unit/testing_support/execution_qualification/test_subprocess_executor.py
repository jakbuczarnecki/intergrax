# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.execution_qualification.contracts import (
    QualificationExecutionContext,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.executor import PytestSubprocessSuiteExecutor


def test_subprocess_smoke_passes_single_file(repo_root: Path, run_artifact_root: Path) -> None:
    suite = QualificationSuite(
        suite_id="smoke-temp-root",
        pytest_arguments=(
            "tests/unit/testing_support/test_pytest_temp_root.py",
            "-q",
            "--tb=no",
        ),
    )
    log_path = run_artifact_root / "smoke-temp-root.log"
    context = QualificationExecutionContext(
        repo_root=repo_root,
        run_artifact_root=run_artifact_root,
        suite_log_path=log_path,
        suite_timeout_seconds=120.0,
        environment_overrides=(),
    )
    result = PytestSubprocessSuiteExecutor().execute(suite, context)
    assert result.status == QualificationSuiteStatus.PASS
    assert result.outcome_kind == QualificationSuiteOutcomeKind.COMPLETED
    assert log_path.is_file()
    assert log_path.stat().st_size > 0


def test_infrastructure_failure_distinct_from_pytest_failure(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    suite = QualificationSuite(
        suite_id="missing-uv",
        pytest_arguments=("tests/unit/testing_support/test_pytest_temp_root.py",),
    )
    context = QualificationExecutionContext(
        repo_root=repo_root,
        run_artifact_root=run_artifact_root,
        suite_log_path=run_artifact_root / "missing.log",
        suite_timeout_seconds=30.0,
        environment_overrides=(),
    )

    class BrokenExecutor(PytestSubprocessSuiteExecutor):
        def execute(self, suite: QualificationSuite, context: QualificationExecutionContext):
            import subprocess

            started = __import__("time").monotonic()
            try:
                subprocess.run(
                    ["__no_such_executable__"],
                    cwd=context.repo_root,
                    timeout=1.0,
                    shell=False,
                )
            except FileNotFoundError:
                duration = __import__("time").monotonic() - started
                from testing_support.execution_qualification.contracts import (
                    ExecutionQualificationSuiteResult,
                )

                return ExecutionQualificationSuiteResult(
                    suite_id=suite.suite_id,
                    command=("__no_such_executable__",),
                    status=QualificationSuiteStatus.FAIL,
                    outcome_kind=QualificationSuiteOutcomeKind.LAUNCH_FAILURE,
                    exit_code=None,
                    duration_seconds=duration,
                    log_path=context.suite_log_path,
                )
            raise AssertionError("expected FileNotFoundError")

    result = BrokenExecutor().execute(suite, context)
    assert result.outcome_kind == QualificationSuiteOutcomeKind.LAUNCH_FAILURE
    assert result.exit_code is None


def test_suite_timeout_terminates_slow_pytest(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    suite = QualificationSuite(
        suite_id="slow",
        pytest_arguments=(
            "tests/unit/testing_support/execution_qualification/test_sleep_gate.py",
            "-q",
            "--tb=no",
        ),
    )
    context = QualificationExecutionContext(
        repo_root=repo_root,
        run_artifact_root=run_artifact_root,
        suite_log_path=run_artifact_root / "slow.log",
        suite_timeout_seconds=1.0,
        environment_overrides=(),
    )
    result = PytestSubprocessSuiteExecutor().execute(suite, context)
    assert result.outcome_kind == QualificationSuiteOutcomeKind.TIMEOUT
    assert result.status == QualificationSuiteStatus.FAIL
