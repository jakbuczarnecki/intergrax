# © Artur Czarnecki. All rights reserved.

"""Repository identity guards for ECAP exact-SHA audits."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepoIdentity:
    remote_url: str
    branch: str
    head_sha: str
    dirty: bool


def assert_git_ancestor(repo_root: Path, *, ancestor_sha: str, descendant_sha: str | None = None) -> None:
    root = repo_root.resolve()
    descendant = descendant_sha or _git(root, "rev-parse", "HEAD")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, descendant],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"{ancestor_sha} is not an ancestor of {descendant}")


def read_repo_identity(repo_root: Path) -> RepoIdentity:
    root = repo_root.resolve()
    head = _git(root, "rev-parse", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain"))
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    remote = _git(root, "remote", "get-url", "origin")
    return RepoIdentity(remote_url=remote, branch=branch, head_sha=head, dirty=dirty)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()
