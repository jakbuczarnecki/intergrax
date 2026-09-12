# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R2 Final — ownership-scoped production drift classification (qualification only)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

R2_IMPLEMENTATION_SHA = "632507420f0ab8360aede43a2740e8fccc44efb4"

# Re-advanced after NPSC-5F/R2 Final Freeze re-sign-off (qualified drift: ``MandatoryEvidencePersistenceError``
# canonicalized to ``intergrax.contracts.execution_evidence.persistence_boundary_errors`` — no journal read change).
R2_POST_QUALIFIED_BASELINE_SHA = "1f7bb7528e8d9b41a0ee00e4831c0f003042474a"

_R2_PROTECTED_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/events/execution_position.py",
        "intergrax/runtime/events/unified_run_journal.py",
        "intergrax/runtime/events/persistence_contract.py",
        "intergrax/runtime/events/__init__.py",
    },
)

_R2_PROTECTED_PATH_PREFIXES: tuple[str, ...] = ("intergrax/runtime/events/stores/",)

# R3 owns redaction/serialization on journal_export; R2 does not file-freeze that module.
_R2_EXPLICITLY_NOT_FILE_FROZEN: frozenset[str] = frozenset(
    {
        "intergrax/runtime/observability/journal_export.py",
    },
)


def _normalize_repo_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def is_r2_protected_production_path(path: str) -> bool:
    """True when ``path`` is an R2-owned journal read / ordering contract surface."""
    normalized = _normalize_repo_path(path)
    if not normalized or not normalized.startswith("intergrax/"):
        return False
    if normalized in _R2_EXPLICITLY_NOT_FILE_FROZEN:
        return False
    if normalized in _R2_PROTECTED_EXACT_PATHS:
        return True
    return normalized.startswith(_R2_PROTECTED_PATH_PREFIXES)


def protected_r2_journal_paths() -> frozenset[str]:
    return _R2_PROTECTED_EXACT_PATHS


def protected_r2_journal_path_prefixes() -> tuple[str, ...]:
    return _R2_PROTECTED_PATH_PREFIXES


def classify_r2_protected_drift(changed_paths: Iterable[str]) -> list[str]:
    drift = {_normalize_repo_path(path) for path in changed_paths if is_r2_protected_production_path(path)}
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


def collect_r2_protected_production_drift(
    repo_root: Path,
    *,
    from_sha: str = R2_IMPLEMENTATION_SHA,
    to_ref: str = "HEAD",
) -> list[str]:
    return classify_r2_protected_drift(
        git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref),
    )
