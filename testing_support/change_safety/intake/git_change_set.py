# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from testing_support.change_safety.discovery.git_paths import git_diff_name_only
from testing_support.change_safety.ports import ChangeSet

_REPO = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class GitChangeSetPort:
    def load(self, *, base_sha: str, head_sha: str) -> ChangeSet:
        return ChangeSet(
            paths=git_diff_name_only(base_sha, head_sha),
            base_sha=base_sha,
            head_sha=head_sha,
        )


def rev_parse_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
