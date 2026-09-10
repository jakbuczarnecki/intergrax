# © Artur Czarnecki. All rights reserved.

"""Execution qualification max_parallel resolution (explicit, ENV, qualified default)."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.execution_qualification.configuration import (
    EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL,
    EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV,
    resolve_execution_qualification_max_parallel,
)
from testing_support.execution_qualification.contracts import QualificationManifestError
from tests.unit.runtime.architecture.npsc5e_r3_final_execution_qualification import (
    NPSC5E_R3_EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL,
    npsc5e_r3_qualification_run_config,
)


def test_env_absent_explicit_none_resolves_to_qualified_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, raising=False)
    assert (
        resolve_execution_qualification_max_parallel(explicit_value=None)
        == EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL
    )
    assert EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL == 2


def test_env_three_explicit_none_resolves_to_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "3")
    assert resolve_execution_qualification_max_parallel(explicit_value=None) == 3


def test_explicit_one_wins_over_env_four(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "4")
    assert resolve_execution_qualification_max_parallel(explicit_value=1) == 1


def test_env_whitespace_stripped_to_three(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, " 3 ")
    assert resolve_execution_qualification_max_parallel(explicit_value=None) == 3


@pytest.mark.parametrize("invalid", ["0", "-1"])
def test_env_non_positive_rejected(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, invalid)
    with pytest.raises(QualificationManifestError):
        resolve_execution_qualification_max_parallel(explicit_value=None)


def test_env_non_integer_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "abc")
    with pytest.raises(QualificationManifestError):
        resolve_execution_qualification_max_parallel(explicit_value=None)


def test_env_decimal_string_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "2.5")
    with pytest.raises(QualificationManifestError):
        resolve_execution_qualification_max_parallel(explicit_value=None)


def test_env_empty_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "")
    with pytest.raises(QualificationManifestError):
        resolve_execution_qualification_max_parallel(explicit_value=None)


def test_env_whitespace_only_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "   ")
    with pytest.raises(QualificationManifestError):
        resolve_execution_qualification_max_parallel(explicit_value=None)


def test_npsc5e_run_config_max_parallel_matches_resolution(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "3")
    config = npsc5e_r3_qualification_run_config(repo_root, run_id="cfg-env-3")
    assert config.max_parallel == 3


def test_npsc5e_explicit_override_wins_over_env(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, "4")
    config = npsc5e_r3_qualification_run_config(
        repo_root,
        run_id="cfg-explicit-1",
        max_parallel=1,
    )
    assert config.max_parallel == 1


def test_qualified_default_constant_is_two() -> None:
    assert NPSC5E_R3_EXECUTION_QUALIFICATION_DEFAULT_MAX_PARALLEL == 2


def test_measured_snapshot_records_resolved_max_parallel_not_only_default(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testing_support.execution_qualification.contracts import (
        QualificationRunManifest,
        QualificationSuite,
    )
    from testing_support.execution_qualification.coordinator import validate_and_run_measured
    from tests.unit.testing_support.execution_qualification.fake_executor import (
        FakeQualificationSuiteExecutor,
    )

    monkeypatch.delenv(EXECUTION_QUALIFICATION_MAX_PARALLEL_ENV, raising=False)
    config = npsc5e_r3_qualification_run_config(
        repo_root,
        run_id="snapshot-mp3",
        max_parallel=3,
    )
    manifest = QualificationRunManifest(
        suites=(QualificationSuite(suite_id="a", pytest_arguments=("x",)),),
    )
    measured = validate_and_run_measured(
        manifest,
        config,
        executor=FakeQualificationSuiteExecutor({}),
    )
    assert config.max_parallel == 3
    assert measured.performance.max_parallel == 3
