# © Artur Czarnecki. All rights reserved.

"""Repository precondition checks for qualification runners."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    HealthVerdict,
    QualificationRepositoryState,
    QualificationRepositoryTransition,
)
from tests.system.functional_diagnostics_h1.qualification_spec import (
    DiagnosticHealthQualificationSpec,
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
    *,
    requires_clean_repository: bool = True,
    requires_origin_development_match: bool = True,
) -> tuple[HealthVerdict, tuple[str, ...]]:
    violations: list[str] = []
    if requires_clean_repository and not state.working_tree_clean:
        violations.append("working_tree_not_clean")
    if requires_origin_development_match and state.head_sha != state.origin_development_sha:
        violations.append(
            "head_not_pushed_to_origin_development:"
            f"head={state.head_sha}:origin={state.origin_development_sha}"
        )
    if violations:
        return HealthVerdict.FAILED_PRECONDITION, tuple(violations)
    return HealthVerdict.PASS, ()


def assert_qualification_repository_postconditions(
    transition: QualificationRepositoryTransition,
    spec: DiagnosticHealthQualificationSpec,
) -> tuple[HealthVerdict, tuple[str, ...]]:
    violations: list[str] = []
    if spec.requires_clean_repository and not transition.end.working_tree_clean:
        violations.append("working_tree_not_clean_at_end")
    if spec.requires_stable_head and transition.start.head_sha != transition.end.head_sha:
        violations.append(
            "head_changed_during_qualification:"
            f"{transition.start.head_sha}->{transition.end.head_sha}"
        )
    if (
        spec.requires_origin_development_match
        and transition.start.origin_development_sha != transition.end.origin_development_sha
    ):
        violations.append(
            "origin_development_changed_during_qualification:"
            f"{transition.start.origin_development_sha}->"
            f"{transition.end.origin_development_sha}"
        )
    if (
        spec.requires_origin_development_match
        and transition.end.head_sha != transition.end.origin_development_sha
    ):
        violations.append(
            "head_not_equal_origin_at_end:"
            f"head={transition.end.head_sha}:origin={transition.end.origin_development_sha}"
        )
    if violations:
        return HealthVerdict.FAILED, tuple(violations)
    return HealthVerdict.PASS, ()
