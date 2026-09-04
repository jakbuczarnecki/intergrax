# © Artur Czarnecki. All rights reserved.

"""Subprocess pytest invocation helpers for H1."""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import PytestSubprocessResult

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COLLECTED_RE = re.compile(r"(\d+) tests? collected|collected (\d+) items?")
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_SKIPPED_RE = re.compile(r"(\d+) skipped")
_XFAILED_RE = re.compile(r"(\d+) xfailed")
_XPASSED_RE = re.compile(r"(\d+) xpassed")
_ERROR_RE = re.compile(r"(\d+) errors?")


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _parse_count(pattern: re.Pattern[str], text: str) -> int:
    match = pattern.search(text)
    if match is None:
        return 0
    for group in match.groups():
        if group is not None:
            return int(group)
    return 0


def run_pytest_subprocess(
    targets: tuple[str, ...],
    *,
    collect_only: bool = False,
    timeout_seconds: float = 900.0,
    extra_args: tuple[str, ...] = (),
    env_overrides: dict[str, str] | None = None,
) -> PytestSubprocessResult:
    command: list[str] = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "--tb=no",
    ]
    if collect_only:
        command.append("--collect-only")
    command.extend(extra_args)
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
            stdout_text = _tail(exc.stdout.decode("utf-8", errors="replace") if exc.stdout else "")
            stderr_text = _tail(exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "")
        else:
            stdout_text = _tail(exc.stdout or "")
            stderr_text = _tail(exc.stderr or "")
        return PytestSubprocessResult(
            exit_code=124,
            collected_count=0,
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
        )
    duration = time.monotonic() - started
    combined = completed.stdout + "\n" + completed.stderr
    collection_errors = combined.lower().count("error collecting")
    if "errors during collection" in combined.lower():
        collection_errors = max(collection_errors, 1)
    return PytestSubprocessResult(
        exit_code=completed.returncode,
        collected_count=_parse_count(_COLLECTED_RE, combined),
        passed=_parse_count(_PASSED_RE, combined),
        failed=_parse_count(_FAILED_RE, combined),
        skipped=_parse_count(_SKIPPED_RE, combined),
        xfailed=_parse_count(_XFAILED_RE, combined),
        xpassed=_parse_count(_XPASSED_RE, combined),
        errors=_parse_count(_ERROR_RE, combined),
        collection_errors=collection_errors,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        duration_seconds=duration,
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
