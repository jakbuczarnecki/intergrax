# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationSuiteResult,
    QualificationCoordinatorError,
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationRunStatus,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator


def _config(repo_root: Path, artifact: Path) -> QualificationRunConfig:
    return QualificationRunConfig(
        repo_root=repo_root,
        max_parallel=1,
        run_artifact_root=artifact,
        suite_timeout_seconds=120.0,
        run_id="hardening-run",
    )


def test_invalid_pass_timeout_combination_rejected() -> None:
    with pytest.raises(ValueError, match="PASS requires COMPLETED"):
        ExecutionQualificationSuiteResult(
            suite_id="x",
            command=(),
            status=QualificationSuiteStatus.PASS,
            outcome_kind=QualificationSuiteOutcomeKind.TIMEOUT,
            exit_code=None,
            duration_seconds=0.0,
            log_path=Path("x.log"),
        )


def test_invalid_pass_nonzero_exit_combination_rejected() -> None:
    with pytest.raises(ValueError, match="PASS requires exit_code 0"):
        ExecutionQualificationSuiteResult(
            suite_id="x",
            command=(),
            status=QualificationSuiteStatus.PASS,
            outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
            exit_code=1,
            duration_seconds=0.0,
            log_path=Path("x.log"),
        )


def test_valid_failure_combinations_still_construct() -> None:
    ExecutionQualificationSuiteResult(
        suite_id="a",
        command=("uv", "run", "pytest"),
        status=QualificationSuiteStatus.FAIL,
        outcome_kind=QualificationSuiteOutcomeKind.PYTEST_NONZERO_EXIT,
        exit_code=1,
        duration_seconds=0.1,
        log_path=Path("a.log"),
    )
    ExecutionQualificationSuiteResult(
        suite_id="b",
        command=(),
        status=QualificationSuiteStatus.FAIL,
        outcome_kind=QualificationSuiteOutcomeKind.LAUNCH_FAILURE,
        exit_code=None,
        duration_seconds=0.0,
        log_path=Path("b.log"),
    )


def test_unexpected_executor_exception_is_coordinator_failure(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    class ExplodingExecutor:
        def execute(self, suite: QualificationSuite, context: object) -> None:
            raise RuntimeError("internal executor bug")

    manifest = QualificationRunManifest(
        suites=(QualificationSuite(suite_id="only", pytest_arguments=("x",)),),
    )
    with pytest.raises(QualificationCoordinatorError, match="unexpected executor failure"):
        QualificationCoordinator(executor=ExplodingExecutor()).run(
            manifest,
            _config(repo_root, run_artifact_root),
        )


def test_wrong_suite_id_from_executor_is_coordinator_failure(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    class WrongIdExecutor:
        def execute(
            self,
            suite: QualificationSuite,
            context: object,
        ) -> ExecutionQualificationSuiteResult:
            return ExecutionQualificationSuiteResult(
                suite_id="not-the-requested-id",
                command=(),
                status=QualificationSuiteStatus.PASS,
                outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
                exit_code=0,
                duration_seconds=0.0,
                log_path=run_artifact_root / "wrong.log",
            )

    manifest = QualificationRunManifest(
        suites=(QualificationSuite(suite_id="expected", pytest_arguments=("x",)),),
    )
    with pytest.raises(QualificationCoordinatorError, match="suite_id"):
        QualificationCoordinator(executor=WrongIdExecutor()).run(
            manifest,
            _config(repo_root, run_artifact_root),
        )


def test_typed_launch_failure_still_collects_all(
    repo_root: Path,
    run_artifact_root: Path,
) -> None:
    class LaunchFailThenPass:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def execute(
            self,
            suite: QualificationSuite,
            context: object,
        ) -> ExecutionQualificationSuiteResult:
            self.seen.append(suite.suite_id)
            if suite.suite_id == "fail-launch":
                return ExecutionQualificationSuiteResult(
                    suite_id=suite.suite_id,
                    command=(),
                    status=QualificationSuiteStatus.FAIL,
                    outcome_kind=QualificationSuiteOutcomeKind.LAUNCH_FAILURE,
                    exit_code=None,
                    duration_seconds=0.0,
                    log_path=run_artifact_root / f"{suite.suite_id}.log",
                )
            return ExecutionQualificationSuiteResult(
                suite_id=suite.suite_id,
                command=(),
                status=QualificationSuiteStatus.PASS,
                outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
                exit_code=0,
                duration_seconds=0.0,
                log_path=run_artifact_root / f"{suite.suite_id}.log",
            )

    executor = LaunchFailThenPass()
    manifest = QualificationRunManifest(
        suites=(
            QualificationSuite(suite_id="fail-launch", pytest_arguments=("a",)),
            QualificationSuite(suite_id="ok", pytest_arguments=("b",)),
        ),
    )
    result = QualificationCoordinator(executor=executor).run(
        manifest,
        _config(repo_root, run_artifact_root),
    )
    assert set(executor.seen) == {"fail-launch", "ok"}
    assert result.status == QualificationRunStatus.FAIL
    assert result.suite_results[0].outcome_kind == QualificationSuiteOutcomeKind.LAUNCH_FAILURE


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def test_subprocess_run_timeout_terminates_direct_child_process(
    run_artifact_root: Path,
) -> None:
    """Same ``subprocess.run(..., timeout=...)`` mechanism as ``PytestSubprocessSuiteExecutor``."""
    import subprocess
    import sys
    import textwrap

    pid_path = run_artifact_root / "direct-child.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_literal = str(pid_path).replace("\\", "\\\\")
    child_script = textwrap.dedent(
        f"""
        import os
        import time
        from pathlib import Path
        Path(r"{pid_literal}").write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(999)
        """
    )
    try:
        subprocess.run(
            [sys.executable, "-c", child_script],
            cwd=run_artifact_root,
            timeout=2.0,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pass
    assert pid_path.is_file()
    recorded_pid = int(pid_path.read_text(encoding="utf-8").strip())
    assert not _process_exists(recorded_pid)
