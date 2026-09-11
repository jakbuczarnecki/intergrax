# © Artur Czarnecki. All rights reserved.

"""Suite executor protocol and default pytest subprocess implementation."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Protocol

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationSuiteResult,
    QualificationExecutionContext,
    QualificationSuite,
    QualificationSuiteOutcomeKind,
    QualificationSuiteStatus,
)


class QualificationSuiteExecutor(Protocol):
    def execute(
        self,
        suite: QualificationSuite,
        context: QualificationExecutionContext,
    ) -> ExecutionQualificationSuiteResult:
        """Run one suite and return a typed result."""
        ...


def build_pytest_command(suite: QualificationSuite) -> tuple[str, ...]:
    return ("uv", "run", "pytest", *suite.pytest_arguments)


def _child_environment(
    base: dict[str, str],
    overrides: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    env = base.copy()
    for key, value in overrides:
        env[key] = value
    return env


class PytestSubprocessSuiteExecutor:
    """Default executor: one ``uv run pytest`` child per suite."""

    def execute(
        self,
        suite: QualificationSuite,
        context: QualificationExecutionContext,
    ) -> ExecutionQualificationSuiteResult:
        command = build_pytest_command(suite)
        log_path = context.suite_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = _child_environment(os.environ, context.environment_overrides)
        started = time.monotonic()
        try:
            with log_path.open("wb") as log_file:
                completed = subprocess.run(
                    list(command),
                    cwd=context.repo_root,
                    env=child_env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=context.suite_timeout_seconds,
                    shell=False,
                    check=False,
                )
        except FileNotFoundError:
            duration = time.monotonic() - started
            return ExecutionQualificationSuiteResult(
                suite_id=suite.suite_id,
                command=command,
                status=QualificationSuiteStatus.FAIL,
                outcome_kind=QualificationSuiteOutcomeKind.LAUNCH_FAILURE,
                exit_code=None,
                duration_seconds=duration,
                log_path=log_path,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - started
            return ExecutionQualificationSuiteResult(
                suite_id=suite.suite_id,
                command=command,
                status=QualificationSuiteStatus.FAIL,
                outcome_kind=QualificationSuiteOutcomeKind.TIMEOUT,
                exit_code=None,
                duration_seconds=duration,
                log_path=log_path,
            )

        duration = time.monotonic() - started
        exit_code = completed.returncode
        if exit_code == 0:
            return ExecutionQualificationSuiteResult(
                suite_id=suite.suite_id,
                command=command,
                status=QualificationSuiteStatus.PASS,
                outcome_kind=QualificationSuiteOutcomeKind.COMPLETED,
                exit_code=exit_code,
                duration_seconds=duration,
                log_path=log_path,
            )
        return ExecutionQualificationSuiteResult(
            suite_id=suite.suite_id,
            command=command,
            status=QualificationSuiteStatus.FAIL,
            outcome_kind=QualificationSuiteOutcomeKind.PYTEST_NONZERO_EXIT,
            exit_code=exit_code,
            duration_seconds=duration,
            log_path=log_path,
        )


def suite_log_path(run_artifact_root: Path, suite_id: str) -> Path:
    """Map a validated suite_id to a log file inside the run artifact root."""
    safe_name = suite_id.replace("/", "_")
    candidate = (run_artifact_root / f"{safe_name}.log").resolve()
    root = run_artifact_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError(f"log path escapes run artifact root for suite {suite_id}") from None
    return candidate
