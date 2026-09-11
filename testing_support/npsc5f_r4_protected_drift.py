# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R4 — ownership-scoped historical reconstruction production drift classification (qualification only)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

R4_IMPLEMENTATION_SHA = "37fb051c7f164d705f628760436b8ea10ee0289f"

_R4_PROTECTED_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "intergrax/contracts/historical_reconstruction.py",
        "intergrax/runtime/observability/historical_reconstruction.py",
        "tests/unit/runtime/architecture/test_npsc5f_r4_reconstruction_asof_bitemporal.py",
        "tests/unit/runtime/architecture/test_npsc5f_r4_final_historical_reconstruction_qualification_and_freeze.py",
        "tests/unit/testing_support/test_npsc5f_r4_final_protected_drift.py",
        "testing_support/npsc5f_r4_protected_drift.py",
        "testing_support/npsc5f_r4_regression_matrix.py",
    },
)

_R4_EXPLICITLY_NOT_FILE_FROZEN_PREFIXES: tuple[str, ...] = (
    "intergrax/runtime/events/",
    "intergrax/contracts/bitemporal_knowledge.py",
    "intergrax/runtime/events/asof_projection.py",
    "intergrax/runtime/diagnostics/execution_reconstruction.py",
)


def _normalize_repo_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def is_r4_protected_production_path(path: str) -> bool:
    """True when ``path`` is an R4-owned reconstruction / as-of composition surface."""
    normalized = _normalize_repo_path(path)
    if not normalized:
        return False
    for prefix in _R4_EXPLICITLY_NOT_FILE_FROZEN_PREFIXES:
        if normalized.startswith(prefix) or normalized == prefix:
            return False
    return normalized in _R4_PROTECTED_EXACT_PATHS


def protected_r4_reconstruction_paths() -> frozenset[str]:
    return _R4_PROTECTED_EXACT_PATHS


def classify_r4_protected_drift(changed_paths: Iterable[str]) -> list[str]:
    drift = {_normalize_repo_path(path) for path in changed_paths if is_r4_protected_production_path(path)}
    return sorted(drift)


def git_changed_paths(
    repo_root: Path,
    *,
    from_sha: str,
    to_ref: str = "HEAD",
) -> list[str]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{from_sha}..{to_ref}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git diff failed ({completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}",
        )
    return [_normalize_repo_path(line) for line in completed.stdout.splitlines() if line.strip()]


def collect_r4_protected_production_drift(
    repo_root: Path,
    *,
    from_sha: str = R4_IMPLEMENTATION_SHA,
    to_ref: str = "origin/development",
) -> list[str]:
    return classify_r4_protected_drift(
        git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref),
    )
