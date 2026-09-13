# © Artur Czarnecki. All rights reserved.

"""Git provenance guards for frozen qualification baseline SHAs (qualification only)."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FrozenBaselineProvenanceError(RuntimeError):
    """Raised when a frozen baseline commit is missing or not reachable from remote refs."""


def _run_git(
    repo_root: Path,
    args: list[str],
    *,
    context: str,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown git error"
        raise FrozenBaselineProvenanceError(f"{context}: {detail}")
    return completed


def assert_frozen_baseline_commit_exists(
    *,
    repo_root: Path,
    baseline_sha: str,
) -> None:
    """Verify ``baseline_sha`` resolves to a commit object in the local object database."""
    _run_git(
        repo_root,
        ["cat-file", "-e", f"{baseline_sha}^{{commit}}"],
        context=f"baseline commit not available from repository refs: {baseline_sha}",
    )


def assert_frozen_baseline_is_ancestor_of_remote(
    *,
    repo_root: Path,
    baseline_sha: str,
    remote_ref: str = "origin/development",
) -> None:
    """Verify ``baseline_sha`` is contained in history reachable from ``remote_ref``."""
    assert_frozen_baseline_commit_exists(repo_root=repo_root, baseline_sha=baseline_sha)
    _run_git(
        repo_root,
        ["merge-base", "--is-ancestor", baseline_sha, remote_ref],
        context=(
            f"baseline commit {baseline_sha} is not an ancestor of {remote_ref} "
            "(not remote-reachable from GitHub development)"
        ),
    )


def assert_frozen_baseline_reachable(
    *,
    repo_root: Path,
    baseline_sha: str,
    remote_ref: str = "origin/development",
) -> None:
    """Existence + remote ancestry — reproducible from ``git fetch`` without reflog."""
    assert_frozen_baseline_is_ancestor_of_remote(
        repo_root=repo_root,
        baseline_sha=baseline_sha,
        remote_ref=remote_ref,
    )
