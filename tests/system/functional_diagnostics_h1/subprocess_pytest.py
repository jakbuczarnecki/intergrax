# © Artur Czarnecki. All rights reserved.

"""Subprocess pytest invocation helpers for H1."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from tests.system.functional_diagnostics_h1.models import (
    PytestFailurePhase,
    PytestSubprocessResult,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_H1_PYTEST_BASETEMP_ROOT = _REPO_ROOT / ".tmp" / "session" / "h1-pytest"
_SHARED_PYTEST_BASETEMP = "build/pytest-basetemp"
_COLLECTED_RE = re.compile(r"(\d+) tests? collected|collected (\d+) items?")
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_SKIPPED_RE = re.compile(r"(\d+) skipped")
_XFAILED_RE = re.compile(r"(\d+) xfailed")
_XPASSED_RE = re.compile(r"(\d+) xpassed")
_ERROR_RE = re.compile(r"(\d+) errors?")
_ERROR_NODE_RE = re.compile(
    r"^ERROR(?:\s+at\s+(?:setup|teardown)\s+of)?\s+(\S+)",
    re.MULTILINE,
)
_FAILED_NODE_RE = re.compile(r"^FAILED\s+(\S+)", re.MULTILINE)


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _parse_count(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def _relative_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _validate_extra_args(extra_args: tuple[str, ...]) -> None:
    for arg in extra_args:
        if arg == "--basetemp" or arg.startswith("--basetemp="):
            raise ValueError("extra_args must not override harness-owned --basetemp")


def allocate_h1_pytest_basetemp() -> tuple[Path, str]:
    """Allocate an invocation-owned H1 pytest basetemp directory."""
    try:
        _H1_PYTEST_BASETEMP_ROOT.mkdir(parents=True, exist_ok=True)
        invocation_id = uuid4().hex
        basetemp = _H1_PYTEST_BASETEMP_ROOT / invocation_id
        basetemp.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise RuntimeError(
            f"failed to allocate H1 pytest basetemp under {_H1_PYTEST_BASETEMP_ROOT.as_posix()}"
        ) from exc
    return basetemp, _relative_repo_path(basetemp)


def build_pytest_subprocess_command(
    targets: tuple[str, ...],
    *,
    basetemp: Path,
    collect_only: bool = False,
    extra_args: tuple[str, ...] = (),
) -> list[str]:
    _validate_extra_args(extra_args)
    command: list[str] = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "--tb=short",
        f"--basetemp={_relative_repo_path(basetemp)}",
    ]
    if collect_only:
        command.append("--collect-only")
    command.extend(extra_args)
    return command


def _extract_test_node_ids(combined: str) -> frozenset[str]:
    node_ids: set[str] = set()
    for pattern in (_ERROR_NODE_RE, _FAILED_NODE_RE):
        node_ids.update(pattern.findall(combined))
    return frozenset(node_ids)


def _derive_collected_count(
    *,
    explicit_collected: int | None,
    passed: int,
    failed: int,
    skipped: int,
    xfailed: int,
    xpassed: int,
    errors: int,
    error_node_ids: frozenset[str],
    collect_only: bool,
) -> int | None:
    if explicit_collected is not None:
        return explicit_collected
    if collect_only:
        return None
    executed = passed + failed + skipped + xfailed + xpassed
    if executed > 0:
        return executed
    if errors > 0 and error_node_ids:
        return len(error_node_ids)
    return None


def infer_pytest_failure_phase(
    combined_output: str,
    *,
    exit_code: int,
) -> PytestFailurePhase | None:
    if exit_code == 0:
        return None
    if exit_code == 124:
        return PytestFailurePhase.TIMEOUT
    lowered = combined_output.lower()
    if "error during collection" in lowered or "errors during collection" in lowered:
        return PytestFailurePhase.COLLECTION
    if "error at setup" in lowered or "errors at setup" in lowered:
        return PytestFailurePhase.SETUP
    if "error at teardown" in lowered or "errors at teardown" in lowered:
        return PytestFailurePhase.TEARDOWN
    if _FAILED_RE.search(combined_output) or " FAILED " in combined_output:
        return PytestFailurePhase.CALL
    if _ERROR_RE.search(combined_output):
        return PytestFailurePhase.UNKNOWN
    return PytestFailurePhase.UNKNOWN


def parse_pytest_subprocess_output(
    stdout: str,
    stderr: str,
    *,
    exit_code: int,
    collect_only: bool = False,
) -> tuple[
    int | None,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    PytestFailurePhase | None,
]:
    combined = stdout + "\n" + stderr
    collection_errors = combined.lower().count("error collecting")
    if "errors during collection" in combined.lower():
        collection_errors = max(collection_errors, 1)
    error_node_ids = _extract_test_node_ids(combined)
    explicit_collected = _parse_count(_COLLECTED_RE, combined)
    passed = _parse_count(_PASSED_RE, combined) or 0
    failed = _parse_count(_FAILED_RE, combined) or 0
    skipped = _parse_count(_SKIPPED_RE, combined) or 0
    xfailed = _parse_count(_XFAILED_RE, combined) or 0
    xpassed = _parse_count(_XPASSED_RE, combined) or 0
    errors = _parse_count(_ERROR_RE, combined) or 0
    collected_count = _derive_collected_count(
        explicit_collected=explicit_collected,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        errors=errors,
        error_node_ids=error_node_ids,
        collect_only=collect_only,
    )
    failure_phase = infer_pytest_failure_phase(combined, exit_code=exit_code)
    return (
        collected_count,
        passed,
        failed,
        skipped,
        xfailed,
        xpassed,
        errors,
        collection_errors,
        failure_phase,
    )


def run_pytest_subprocess(
    targets: tuple[str, ...],
    *,
    collect_only: bool = False,
    timeout_seconds: float = 900.0,
    extra_args: tuple[str, ...] = (),
    env_overrides: dict[str, str] | None = None,
) -> PytestSubprocessResult:
    basetemp, basetemp_path = allocate_h1_pytest_basetemp()
    command = build_pytest_subprocess_command(
        targets,
        basetemp=basetemp,
        collect_only=collect_only,
        extra_args=extra_args,
    )
    started = time.monotonic()
    env = None
    if env_overrides:
        import os

        env = os.environ.copy()
        env.update(env_overrides)
    try:
        completed = subprocess.run(
            command,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - started
        if isinstance(exc.stdout, bytes):
            stdout_text = _tail(
                exc.stdout.decode("utf-8", errors="replace") if exc.stdout else ""
            )
            stderr_text = _tail(
                exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            )
        else:
            stdout_text = _tail(exc.stdout or "")
            stderr_text = _tail(exc.stderr or "")
        return PytestSubprocessResult(
            exit_code=124,
            collected_count=None,
            passed=0,
            failed=0,
            skipped=0,
            xfailed=0,
            xpassed=0,
            errors=0,
            collection_errors=0,
            stdout_tail=stdout_text,
            stderr_tail=stderr_text + "\nFAILED_TIMEOUT",
            duration_seconds=duration,
            basetemp_path=basetemp_path,
            failure_phase=PytestFailurePhase.TIMEOUT,
        )
    duration = time.monotonic() - started
    (
        collected_count,
        passed,
        failed,
        skipped,
        xfailed,
        xpassed,
        errors,
        collection_errors,
        failure_phase,
    ) = parse_pytest_subprocess_output(
        completed.stdout,
        completed.stderr,
        exit_code=completed.returncode,
        collect_only=collect_only,
    )
    return PytestSubprocessResult(
        exit_code=completed.returncode,
        collected_count=collected_count,
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        xpassed=xpassed,
        errors=errors,
        collection_errors=collection_errors,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        duration_seconds=duration,
        basetemp_path=basetemp_path,
        failure_phase=failure_phase,
    )


def classify_pytest_exit(result: PytestSubprocessResult, *, collect_only: bool = False) -> str:
    if result.exit_code == 124:
        return "FAILED_TIMEOUT"
    if collect_only:
        if result.collection_errors > 0 or result.exit_code not in {0, 5}:
            return "FAILED"
        return "PASS"
    if result.failed > 0 or result.errors > 0 or result.exit_code not in {0, 5}:
        return "FAILED"
    if result.xpassed > 0:
        return "FAILED"
    return "PASS"


def h1_pytest_basetemp_root() -> Path:
    return _H1_PYTEST_BASETEMP_ROOT


def shared_pytest_basetemp_marker() -> str:
    return _SHARED_PYTEST_BASETEMP
