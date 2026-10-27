#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Git path helpers for diff-scoped CI gates."""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
    )


def _lines(result: subprocess.CompletedProcess[str]) -> list[str]:
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def staged_paths() -> frozenset[str]:
    return frozenset(_lines(_run_git("diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB")))


def push_paths() -> frozenset[str]:
    upstream = _run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream.returncode != 0:
        return frozenset(_lines(_run_git("diff", "--name-only", "HEAD")))
    base = upstream.stdout.strip()
    merge_base = _run_git("merge-base", "HEAD", base)
    if merge_base.returncode != 0:
        return frozenset(_lines(_run_git("diff", "--name-only", "HEAD")))
    return frozenset(_lines(_run_git("diff", "--name-only", f"{merge_base.stdout.strip()}..HEAD")))


def paths_touch_prefix(paths: frozenset[str], prefixes: tuple[str, ...]) -> bool:
    for path in paths:
        normalized = path.replace("\\", "/")
        for prefix in prefixes:
            if normalized.startswith(prefix):
                return True
    return False
