# © Artur Czarnecki. All rights reserved.

"""Unit tests for H1 subprocess pytest isolation and output parsing."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.system.functional_diagnostics_h1.local_integration import (
    LOCAL_INTEGRATION_ENV_OVERRIDES,
    evaluate_local_integration_suite,
)
from tests.system.functional_diagnostics_h1.models import PytestFailurePhase, PytestSubprocessResult
from tests.system.functional_diagnostics_h1.subprocess_pytest import (
    allocate_h1_pytest_basetemp,
    build_pytest_subprocess_command,
    classify_pytest_exit,
    h1_pytest_basetemp_root,
    parse_pytest_subprocess_output,
    run_pytest_subprocess,
    shared_pytest_basetemp_marker,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PASS_FIXTURE = (
    "tests/unit/system/functional_diagnostics_h1/fixtures/synthetic_pass_a.py"
)


def _completed_process(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["pytest"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_sequential_invocations_allocate_distinct_basetemp_paths() -> None:
    first, first_relative = allocate_h1_pytest_basetemp()
    second, second_relative = allocate_h1_pytest_basetemp()
    assert first != second
    assert first_relative != second_relative
    assert first_relative.startswith(".tmp/session/h1-pytest/")
    assert second_relative.startswith(".tmp/session/h1-pytest/")


def test_command_uses_isolated_h1_basetemp_not_shared_global() -> None:
    basetemp, relative = allocate_h1_pytest_basetemp()
    command = build_pytest_subprocess_command(
        ("tests/example.py",),
        basetemp=basetemp,
    )
    command_text = " ".join(command)
    assert f"--basetemp={relative}" in command_text
    assert shared_pytest_basetemp_marker() not in command_text
    assert "--tb=short" in command


def test_extra_args_basetemp_override_rejected() -> None:
    basetemp, _relative = allocate_h1_pytest_basetemp()
    with pytest.raises(ValueError, match="basetemp"):
        build_pytest_subprocess_command(
            ("tests/example.py",),
            basetemp=basetemp,
            extra_args=("--basetemp=build/pytest-basetemp",),
        )


def test_failure_preserves_evidence_and_basetemp() -> None:
    basetemp, relative = allocate_h1_pytest_basetemp()
    stdout = (
        "ERROR at setup of tests/foo.py::test_bar\n"
        "E   PermissionError: [WinError 32] file in use\n"
        "1 error in 0.12s"
    )
    with patch(
        "tests.system.functional_diagnostics_h1.subprocess_pytest.allocate_h1_pytest_basetemp",
        return_value=(basetemp, relative),
    ), patch(
        "tests.system.functional_diagnostics_h1.subprocess_pytest.subprocess.run",
        return_value=_completed_process(returncode=1, stdout=stdout, stderr="setup failed"),
    ):
        result = run_pytest_subprocess(("tests/foo.py",))
    suite = evaluate_local_integration_suite("tests/foo.py", result)
    assert classify_pytest_exit(result) == "FAILED"
    assert result.basetemp_path == relative
    assert "PermissionError" in result.stdout_tail
    assert result.stderr_tail == "setup failed"
    assert result.failure_phase is PytestFailurePhase.SETUP
    assert suite.failure_evidence is not None
    assert suite.failure_evidence.basetemp_path == relative


def test_pass_result_parsing() -> None:
    stdout = "2 passed in 0.05s"
    collected, passed, failed, skipped, xfailed, xpassed, errors, collection_errors, phase = (
        parse_pytest_subprocess_output(stdout, "", exit_code=0)
    )
    result = PytestSubprocessResult(
        exit_code=0,
        collected_count=collected,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        errors=errors,
        collection_errors=collection_errors,
        stdout_tail=stdout,
        stderr_tail="",
        duration_seconds=0.1,
    )
    assert classify_pytest_exit(result) == "PASS"
    assert passed == 2
    assert collected == 2
    assert errors == 0
    assert phase is None


def test_terminal_style_passed_with_error_is_failed() -> None:
    stdout = "12 passed, 1 error in 1.23s"
    collected, passed, failed, skipped, xfailed, xpassed, errors, collection_errors, phase = (
        parse_pytest_subprocess_output(stdout, "", exit_code=1)
    )
    result = PytestSubprocessResult(
        exit_code=1,
        collected_count=collected,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        errors=errors,
        collection_errors=collection_errors,
        stdout_tail=stdout,
        stderr_tail="",
        duration_seconds=0.1,
    )
    assert classify_pytest_exit(result) == "FAILED"
    assert passed == 12
    assert errors == 1
    assert collected == 12


def test_error_only_case_fails_without_false_collected() -> None:
    stdout = (
        "ERROR tests/foo.py::test_a\n"
        "ERROR tests/foo.py::test_b\n"
        "2 errors in 0.50s"
    )
    collected, passed, failed, skipped, xfailed, xpassed, errors, collection_errors, phase = (
        parse_pytest_subprocess_output(stdout, "", exit_code=1)
    )
    result = PytestSubprocessResult(
        exit_code=1,
        collected_count=collected,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        errors=errors,
        collection_errors=collection_errors,
        stdout_tail=stdout,
        stderr_tail="",
        duration_seconds=0.1,
    )
    assert classify_pytest_exit(result) == "FAILED"
    assert passed == 0
    assert errors == 2
    assert collected == 2


def test_timeout_preserves_basetemp_and_failed_timeout() -> None:
    basetemp, relative = allocate_h1_pytest_basetemp()
    with patch(
        "tests.system.functional_diagnostics_h1.subprocess_pytest.allocate_h1_pytest_basetemp",
        return_value=(basetemp, relative),
    ), patch(
        "tests.system.functional_diagnostics_h1.subprocess_pytest.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1.0, output="partial"),
    ):
        result = run_pytest_subprocess(("tests/foo.py",), timeout_seconds=1.0)
    assert result.exit_code == 124
    assert classify_pytest_exit(result) == "FAILED_TIMEOUT"
    assert result.basetemp_path == relative
    assert result.failure_phase is PytestFailurePhase.TIMEOUT
    assert "FAILED_TIMEOUT" in result.stderr_tail


def test_env_overrides_forwarded_to_subprocess() -> None:
    captured: dict[str, object] = {}

    def _capture_run(*_args, **kwargs) -> subprocess.CompletedProcess[str]:
        captured["env"] = kwargs.get("env")
        return _completed_process(stdout="1 passed in 0.01s")

    with patch(
        "tests.system.functional_diagnostics_h1.subprocess_pytest.subprocess.run",
        side_effect=_capture_run,
    ):
        run_pytest_subprocess(
            (_PASS_FIXTURE,),
            env_overrides=LOCAL_INTEGRATION_ENV_OVERRIDES,
        )
    env = captured["env"]
    assert isinstance(env, dict)
    assert (
        env["INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"]
        == LOCAL_INTEGRATION_ENV_OVERRIDES[
            "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET"
        ]
    )


def test_extra_args_preserved_without_basetemp_override() -> None:
    basetemp, _relative = allocate_h1_pytest_basetemp()
    command = build_pytest_subprocess_command(
        ("tests/example.py",),
        basetemp=basetemp,
        extra_args=("-k", "test_name"),
    )
    assert "-k" in command
    assert "test_name" in command


def test_shared_basetemp_contamination_does_not_affect_h1_namespace() -> None:
    shared_root = _REPO_ROOT / shared_pytest_basetemp_marker()
    shared_root.mkdir(parents=True, exist_ok=True)
    locked = shared_root / "foreign_locked.sqlite"
    locked.write_text("locked", encoding="utf-8")
    h1_root = h1_pytest_basetemp_root()
    assert h1_root != shared_root
    basetemp, relative = allocate_h1_pytest_basetemp()
    assert not relative.startswith(shared_pytest_basetemp_marker())
    assert basetemp.is_dir()
    command = build_pytest_subprocess_command((_PASS_FIXTURE,), basetemp=basetemp)
    assert shared_pytest_basetemp_marker() not in " ".join(command)


def test_run_pytest_subprocess_sequential_calls_use_distinct_basetemp() -> None:
    captured_basetemps: list[str] = []

    def _capture_run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        basetemp_arg = next(arg for arg in command if arg.startswith("--basetemp="))
        captured_basetemps.append(basetemp_arg)
        return _completed_process(stdout="1 passed in 0.01s")

    with patch(
        "tests.system.functional_diagnostics_h1.subprocess_pytest.subprocess.run",
        side_effect=_capture_run,
    ):
        run_pytest_subprocess((_PASS_FIXTURE,))
        run_pytest_subprocess((_PASS_FIXTURE,))
    assert len(captured_basetemps) == 2
    assert captured_basetemps[0] != captured_basetemps[1]
    assert all(".tmp/session/h1-pytest/" in item for item in captured_basetemps)
    assert all(shared_pytest_basetemp_marker() not in item for item in captured_basetemps)
