# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R3 Final — ownership-scoped export-security production drift classification (qualification only)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

R3_IMPLEMENTATION_SHA = "0346face3ef68d8f21504822a26f8f45f2384cf9"

_R3_PROTECTED_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/observability/export_boundary.py",
        "intergrax/runtime/observability/journal_export.py",
        "intergrax/runtime/observability/export_bridge.py",
    },
)

# R4 and adjacent observability modules remain editable without R3 file-freeze.
_R3_EXPLICITLY_NOT_FILE_FROZEN_PREFIXES: tuple[str, ...] = (
    "intergrax/runtime/observability/export_routing.py",
    "intergrax/runtime/observability/causal_evidence",
    "intergrax/runtime/observability/persistence_conformance.py",
)


def _normalize_repo_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def is_r3_protected_production_path(path: str) -> bool:
    """True when ``path`` is an R3-owned safe export / journal export contract surface."""
    normalized = _normalize_repo_path(path)
    if not normalized or not normalized.startswith("intergrax/runtime/observability/"):
        return False
    for prefix in _R3_EXPLICITLY_NOT_FILE_FROZEN_PREFIXES:
        if normalized.startswith(prefix) or normalized == prefix:
            return False
    if normalized in _R3_PROTECTED_EXACT_PATHS:
        return True
    return False


def protected_r3_export_paths() -> frozenset[str]:
    return _R3_PROTECTED_EXACT_PATHS


def classify_r3_protected_drift(changed_paths: Iterable[str]) -> list[str]:
    drift = {_normalize_repo_path(path) for path in changed_paths if is_r3_protected_production_path(path)}
    return sorted(drift)


def git_changed_paths(
    repo_root: Path,
    *,
    from_sha: str,
    to_ref: str = "HEAD",
) -> list[str]:
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{from_sha}..{to_ref}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git diff failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}",
        )
    return [_normalize_repo_path(line) for line in proc.stdout.splitlines() if line.strip()]


def collect_r3_protected_production_drift(
    repo_root: Path,
    *,
    from_sha: str = R3_IMPLEMENTATION_SHA,
    to_ref: str = "origin/development",
) -> list[str]:
    return classify_r3_protected_drift(
        git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref),
    )
