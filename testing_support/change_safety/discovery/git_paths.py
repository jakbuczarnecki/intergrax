# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def git_ls_files(prefix: str) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files", prefix],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ()
    return tuple(line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip())


def git_diff_name_only(base_sha: str, head_sha: str) -> frozenset[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return frozenset()
    return frozenset(line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip())
