# © Artur Czarnecki. All rights reserved.

"""Repository precondition checks for qualification runners."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    HealthVerdict,
    QualificationRepositoryState,
)


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def capture_qualification_repository_state(repo_root: Path) -> QualificationRepositoryState:
    porcelain = _run_git(repo_root, "status", "--porcelain")
    return QualificationRepositoryState(
        head_sha=_run_git(repo_root, "rev-parse", "HEAD"),
        origin_development_sha=_run_git(repo_root, "rev-parse", "origin/development"),
        working_tree_clean=porcelain == "",
    )


def assert_qualification_repository_state(
    state: QualificationRepositoryState,
) -> tuple[HealthVerdict, tuple[str, ...]]:
    violations: list[str] = []
    if not state.working_tree_clean:
        violations.append("working_tree_not_clean")
    if state.head_sha != state.origin_development_sha:
        violations.append(
            f"head_not_pushed_to_origin_development:head={state.head_sha}:origin={state.origin_development_sha}"
        )
    if violations:
        return HealthVerdict.FAILED_PRECONDITION, tuple(violations)
    return HealthVerdict.PASS, ()
