# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunManifest,
    QualificationSuite,
)


def test_duplicate_suite_ids_rejected(repo_root: Path, run_artifact_root: Path) -> None:
    suite = QualificationSuite(
        suite_id="dup",
        pytest_arguments=("tests/unit/testing_support/test_pytest_temp_root.py",),
    )
    with pytest.raises(ValueError, match="duplicate"):
        QualificationRunManifest(suites=(suite, suite))


def test_empty_pytest_arguments_rejected() -> None:
    with pytest.raises(ValueError, match="pytest_arguments"):
        QualificationSuite(suite_id="empty", pytest_arguments=())


def test_invalid_suite_id_rejected() -> None:
    with pytest.raises(ValueError, match="suite_id"):
        QualificationSuite(suite_id="../bad", pytest_arguments=("x",))


def test_invalid_max_parallel_rejected(repo_root: Path, run_artifact_root: Path) -> None:
    with pytest.raises(ValueError, match="max_parallel"):
        QualificationRunConfig(
            repo_root=repo_root,
            max_parallel=0,
            run_artifact_root=run_artifact_root,
            suite_timeout_seconds=60.0,
            run_id="bad-parallel",
        )
