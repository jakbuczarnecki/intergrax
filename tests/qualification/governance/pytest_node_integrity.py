# © Artur Czarnecki. All rights reserved.

"""Reusable pytest node-id collection integrity helpers for qualification catalogs."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


def run_pytest_collect_only_q(
    node_ids: Sequence[str],
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *node_ids],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def parse_collect_only_q(stdout: str) -> frozenset[str]:
    collected: set[str] = set()
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or " collected in " in stripped:
            continue
        if "::" in stripped:
            collected.add(stripped)
    return frozenset(collected)


def pytest_nodes_missing_from_collection(
    node_ids: Sequence[str],
    repo_root: Path,
) -> tuple[frozenset[str], subprocess.CompletedProcess[str]]:
    if not node_ids:
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="",
        )
        return frozenset(), proc
    proc = run_pytest_collect_only_q(node_ids, repo_root)
    collected = parse_collect_only_q(proc.stdout)
    missing = frozenset(node_id for node_id in node_ids if node_id not in collected)
    return missing, proc


def pytest_nodes_are_collectable(
    node_ids: Sequence[str],
    repo_root: Path,
) -> bool:
    missing, proc = pytest_nodes_missing_from_collection(node_ids, repo_root)
    return proc.returncode == 0 and not missing


def run_pytest_execute_q(
    node_ids: Sequence[str],
    repo_root: Path,
) -> subprocess.CompletedProcess[str]:
    if not node_ids:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0 passed in 0.00s",
            stderr="",
        )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:xdist",
            "-q",
            "--tb=short",
            *node_ids,
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass(frozen=True, slots=True)
class PytestExecuteSummary:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    xfailed: int = 0
    errors: int = 0


_SUMMARY_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|xfailed|error|errors)")


def parse_pytest_execute_q_summary(stdout: str, stderr: str) -> PytestExecuteSummary:
    combined = f"{stdout}\n{stderr}"
    passed = failed = skipped = xfailed = errors = 0
    for line in reversed(combined.splitlines()):
        if " in " in line and any(
            token in line for token in ("passed", "failed", "skipped", "error")
        ):
            for count, label in _SUMMARY_COUNT_RE.findall(line):
                value = int(count)
                if label == "passed":
                    passed = value
                elif label == "failed":
                    failed = value
                elif label == "skipped":
                    skipped = value
                elif label == "xfailed":
                    xfailed = value
                elif label in ("error", "errors"):
                    errors = value
            break
    return PytestExecuteSummary(
        passed=passed,
        failed=failed,
        skipped=skipped,
        xfailed=xfailed,
        errors=errors,
    )
