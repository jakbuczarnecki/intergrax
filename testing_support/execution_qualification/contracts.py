# © Artur Czarnecki. All rights reserved.

"""Typed contracts for bounded parallel execution qualification (R1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_SAFE_SUITE_ID_RE = re.compile(r"^[\w.-]+$")
_SAFE_RESOURCE_ID_RE = re.compile(r"^[\w./-]+$")
_SAFE_RUN_ID_RE = re.compile(r"^[\w.-]+$")


class QualificationSuiteStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class QualificationSuiteOutcomeKind(StrEnum):
    """How the suite run concluded — distinct from pytest assertion failure."""

    COMPLETED = "COMPLETED"
    PYTEST_NONZERO_EXIT = "PYTEST_NONZERO_EXIT"
    LAUNCH_FAILURE = "LAUNCH_FAILURE"
    TIMEOUT = "TIMEOUT"
    NOT_STARTED = "NOT_STARTED"


class QualificationRunStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class QualificationSuite:
    """One isolated pytest subprocess definition."""

    suite_id: str
    pytest_arguments: tuple[str, ...]
    exclusive_resource_id: str | None = None
    environment_overrides: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.suite_id or not _SAFE_SUITE_ID_RE.fullmatch(self.suite_id):
            raise ValueError(f"invalid suite_id: {self.suite_id!r}")
        if not self.pytest_arguments:
            raise ValueError(f"suite {self.suite_id}: pytest_arguments must be non-empty")
        if self.exclusive_resource_id is not None:
            rid = self.exclusive_resource_id
            if not rid or not _SAFE_RESOURCE_ID_RE.fullmatch(rid):
                raise ValueError(f"suite {self.suite_id}: invalid exclusive_resource_id")
            if ".." in rid:
                raise ValueError(f"suite {self.suite_id}: exclusive_resource_id must not contain '..'")


@dataclass(frozen=True, slots=True)
class QualificationRunManifest:
    suites: tuple[QualificationSuite, ...]

    def __post_init__(self) -> None:
        if not self.suites:
            raise ValueError("manifest must contain at least one suite")
        seen: set[str] = set()
        for suite in self.suites:
            if suite.suite_id in seen:
                raise ValueError(f"duplicate suite_id: {suite.suite_id}")
            seen.add(suite.suite_id)


@dataclass(frozen=True, slots=True)
class QualificationRunConfig:
    repo_root: Path
    max_parallel: int
    run_artifact_root: Path
    suite_timeout_seconds: float
    run_id: str

    def __post_init__(self) -> None:
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if self.suite_timeout_seconds <= 0:
            raise ValueError("suite_timeout_seconds must be positive")
        if not self.run_id or not _SAFE_RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError("run_id must be a path-safe identifier")
        resolved_root = self.run_artifact_root.resolve()
        resolved_repo = self.repo_root.resolve()
        try:
            resolved_root.relative_to(resolved_repo)
        except ValueError:
            raise ValueError("run_artifact_root must be under repo_root") from None


@dataclass(frozen=True, slots=True)
class QualificationExecutionContext:
    repo_root: Path
    run_artifact_root: Path
    suite_log_path: Path
    suite_timeout_seconds: float
    environment_overrides: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ExecutionQualificationSuiteResult:
    suite_id: str
    command: tuple[str, ...]
    status: QualificationSuiteStatus
    outcome_kind: QualificationSuiteOutcomeKind
    exit_code: int | None
    duration_seconds: float
    log_path: Path

    def __post_init__(self) -> None:
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.status == QualificationSuiteStatus.PASS:
            if self.outcome_kind is not QualificationSuiteOutcomeKind.COMPLETED:
                raise ValueError(
                    f"PASS requires COMPLETED outcome, got {self.outcome_kind}"
                )
            if self.exit_code != 0:
                raise ValueError("PASS requires exit_code 0")
        if self.outcome_kind is QualificationSuiteOutcomeKind.COMPLETED:
            if self.status is not QualificationSuiteStatus.PASS:
                raise ValueError("COMPLETED requires PASS status")
            if self.exit_code != 0:
                raise ValueError("COMPLETED requires exit_code 0")
        if self.outcome_kind is QualificationSuiteOutcomeKind.PYTEST_NONZERO_EXIT:
            if self.status is not QualificationSuiteStatus.FAIL:
                raise ValueError("PYTEST_NONZERO_EXIT requires FAIL status")
            if self.exit_code is None or self.exit_code == 0:
                raise ValueError("PYTEST_NONZERO_EXIT requires nonzero exit_code")
        if self.outcome_kind is QualificationSuiteOutcomeKind.TIMEOUT:
            if self.status is not QualificationSuiteStatus.FAIL:
                raise ValueError("TIMEOUT requires FAIL status")
            if self.exit_code == 0:
                raise ValueError("TIMEOUT cannot have exit_code 0")
        if self.outcome_kind is QualificationSuiteOutcomeKind.LAUNCH_FAILURE:
            if self.status is not QualificationSuiteStatus.FAIL:
                raise ValueError("LAUNCH_FAILURE requires FAIL status")
            if self.exit_code == 0:
                raise ValueError("LAUNCH_FAILURE cannot have exit_code 0")
        if self.outcome_kind is QualificationSuiteOutcomeKind.NOT_STARTED:
            if self.status is not QualificationSuiteStatus.SKIP:
                raise ValueError("NOT_STARTED requires SKIP status")


@dataclass(frozen=True, slots=True)
class ExecutionQualificationRunResult:
    run_id: str
    status: QualificationRunStatus
    suite_results: tuple[ExecutionQualificationSuiteResult, ...]


class QualificationManifestError(ValueError):
    """Invalid manifest or run configuration detected before any child launch."""


class QualificationCoordinatorError(RuntimeError):
    """Coordinator cannot continue the run reliably."""
