# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from testing_support.change_safety.ports import ChangeSet

_REPO = Path(__file__).resolve().parents[3]


def parse_name_status(output: str) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...], frozenset[str]]:
    """Parse ``git diff --name-status -M`` (tab separated). Renames contribute both old and new path."""
    statuses: list[tuple[str, str]] = []
    renames: list[tuple[str, str]] = []
    paths: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0][0]
        if code in {"R", "C"} and len(parts) >= 3:
            old, new = parts[1].replace("\\", "/"), parts[2].replace("\\", "/")
            renames.append((old, new))
            statuses.extend([(old, "D"), (new, "A")])
            paths.update({old, new})
        elif len(parts) >= 2:
            path = parts[1].replace("\\", "/")
            statuses.append((path, code))
            paths.add(path)
    return tuple(sorted(statuses)), tuple(sorted(renames)), frozenset(paths)


@dataclass(frozen=True)
class GitChangeSetPort:
    def load(self, *, base_sha: str, head_sha: str) -> ChangeSet:
        result = subprocess.run(
            ["git", "diff", "--name-status", "-M", f"{base_sha}..{head_sha}"],
            cwd=_REPO,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git diff failed for {base_sha}..{head_sha}: {result.stderr.strip()}")
        statuses, renames, paths = parse_name_status(result.stdout)
        return ChangeSet(paths=paths, base_sha=base_sha, head_sha=head_sha, statuses=statuses, renames=renames)


def rev_parse_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
