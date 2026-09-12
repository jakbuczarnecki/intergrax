# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R1 Final — ownership-scoped production drift classification (qualification only)."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

R1_IMPLEMENTATION_SHA = "455d3b216f0ad56ea9cdf9db6e0f760b50063a81"

# Qualified post-R2 enum extension: ``RuntimeEventType.EXECUTION_FAILED`` (40cc8c11e).
# Advanced after NPSC-5F/R1 event-spine drift reconciliation on integrated ``development``.
R1_POST_R2_QUALIFIED_BASELINE_SHA = "ec23b24d4bb398c83ed310c9deee7d462d73a7e0"

_R1_PROTECTED_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "intergrax/runtime/events/evidence_durability.py",
        "intergrax/runtime/events/event_bus.py",
        "intergrax/runtime/events/persistence_contract.py",
        "intergrax/runtime/events/runtime_event.py",
    },
)

_R1_PROTECTED_PATH_PREFIXES: tuple[str, ...] = ("intergrax/runtime/events/stores/",)


def _normalize_repo_path(path: str) -> str:
    return path.strip().replace("\\", "/")


def is_r1_protected_production_path(path: str) -> bool:
    """True when ``path`` is an R1-owned evidence durability / tenant contract surface."""
    normalized = _normalize_repo_path(path)
    if not normalized or not normalized.startswith("intergrax/"):
        return False
    if normalized in _R1_PROTECTED_EXACT_PATHS:
        return True
    return normalized.startswith(_R1_PROTECTED_PATH_PREFIXES)


def protected_r1_evidence_paths() -> frozenset[str]:
    """Documented R1 protected production paths (exact files + stores/ prefix rule)."""
    return _R1_PROTECTED_EXACT_PATHS


def protected_r1_evidence_path_prefixes() -> tuple[str, ...]:
    return _R1_PROTECTED_PATH_PREFIXES


def classify_r1_protected_drift(changed_paths: Iterable[str]) -> list[str]:
    """Return sorted R1-protected production paths present in ``changed_paths``."""
    drift = {_normalize_repo_path(path) for path in changed_paths if is_r1_protected_production_path(path)}
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


def collect_r1_protected_production_drift(
    repo_root: Path,
    *,
    from_sha: str = R1_IMPLEMENTATION_SHA,
    to_ref: str = "HEAD",
) -> list[str]:
    """Paths under R1 evidence contract ownership changed between ``from_sha`` and ``to_ref``."""
    return classify_r1_protected_drift(
        git_changed_paths(repo_root, from_sha=from_sha, to_ref=to_ref),
    )
