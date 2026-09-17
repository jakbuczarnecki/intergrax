# © Artur Czarnecki. All rights reserved.

"""Reusable pytest node-id collection integrity helpers for qualification catalogs."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
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
