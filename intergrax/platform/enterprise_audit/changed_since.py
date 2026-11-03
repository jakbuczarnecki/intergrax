# © Artur Czarnecki. All rights reserved.

"""CHANGED-SINCE check selection via git diff + invariant floor."""

from __future__ import annotations

import subprocess
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import ManifestCheck
from intergrax.platform.enterprise_audit.impact_map import (
    load_default_impact_map,
    selected_check_ids_from_impact,
    unmapped_fail_closed_paths,
)
from intergrax.platform.enterprise_audit.repo_identity import assert_git_ancestor


def git_changed_files(repo_root: Path, since_sha: str) -> frozenset[str]:
    root = repo_root.resolve()
    completed = subprocess.run(
        ["git", "diff", "--name-only", f"{since_sha}..HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = {line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()}
    return frozenset(paths)


def _entrypoint_paths(entrypoint: str) -> tuple[str, ...]:
    if "::" in entrypoint:
        file_part = entrypoint.split("::", 1)[0]
        return (file_part,)
    return (entrypoint,)


def _check_touched(check: ManifestCheck, changed: frozenset[str]) -> bool:
    for ep_path in _entrypoint_paths(check.entrypoint):
        normalized = ep_path.replace("\\", "/")
        for path in changed:
            if path == normalized or path.startswith(normalized.rstrip("/") + "/"):
                return True
            if normalized.startswith(path.rstrip("/") + "/") or normalized == path:
                return True
    return False


def invariant_floor(checks: tuple[ManifestCheck, ...]) -> frozenset[str]:
    floor: set[str] = set()
    for check in checks:
        if getattr(check, "fast_mandatory", False):
            floor.add(check.check_id)
        if check.check_id.startswith("TEN-") or check.check_id.startswith("VBN-"):
            floor.add(check.check_id)
    return frozenset(floor)


_EVOL_PREFIX = "tests/qualification/evol_x/"


def _evol_entrypoint_paths(check: ManifestCheck) -> tuple[str, ...]:
    paths = []
    for ep_path in _entrypoint_paths(check.entrypoint):
        normalized = ep_path.replace("\\", "/")
        if _EVOL_PREFIX in normalized or normalized.startswith(_EVOL_PREFIX):
            paths.append(normalized)
    return tuple(paths)


def evol_checks_by_path(checks: tuple[ManifestCheck, ...]) -> dict[str, frozenset[str]]:
    mapping: dict[str, set[str]] = {}
    for check in checks:
        for ep in _evol_entrypoint_paths(check):
            mapping.setdefault(ep, set()).add(check.check_id)
        if check.check_id.startswith("ENT-A-EVOL") or check.check_id.startswith("ENT-A-INV"):
            for ep in _entrypoint_paths(check.entrypoint):
                mapping.setdefault(ep.replace("\\", "/"), set()).add(check.check_id)
    return {k: frozenset(v) for k, v in mapping.items()}


def unmapped_evol_changed_paths(
    checks: tuple[ManifestCheck, ...],
    changed: frozenset[str],
) -> frozenset[str]:
    """Paths under evol_x/ changed but not mapped to any manifest check (fail-closed)."""
    mapping = evol_checks_by_path(checks)
    unmapped: set[str] = set()
    for path in changed:
        if not path.startswith(_EVOL_PREFIX):
            continue
        matched = False
        for ep, _ids in mapping.items():
            if path == ep or path.startswith(ep.rstrip("/") + "/"):
                matched = True
                break
        if not matched:
            unmapped.add(path)
    return frozenset(unmapped)


def select_checks_for_changed_since(
    checks: tuple[ManifestCheck, ...],
    *,
    repo_root: Path,
    since_sha: str,
) -> frozenset[str]:
    assert_git_ancestor(repo_root, ancestor_sha=since_sha)
    changed = git_changed_files(repo_root, since_sha)
    unmapped = unmapped_evol_changed_paths(checks, changed)
    if unmapped:
        raise ValueError("CHANGED-SINCE fail-closed: unmapped evol_x paths: " + ", ".join(sorted(unmapped)[:8]))
    impact_prefixes = load_default_impact_map()
    entrypoint_covered = frozenset(
        path for path in changed if any(_check_touched(check, frozenset({path})) for check in checks)
    )
    platform_unmapped = unmapped_fail_closed_paths(
        changed,
        prefixes=impact_prefixes,
        entrypoint_covered=entrypoint_covered,
    )
    if platform_unmapped:
        sample = ", ".join(sorted(platform_unmapped)[:8])
        raise ValueError(f"CHANGED-SINCE fail-closed: unmapped platform paths: {sample}")
    selected: set[str] = set(invariant_floor(checks))
    selected.update(selected_check_ids_from_impact(changed, prefixes=impact_prefixes))
    for check in checks:
        if _check_touched(check, changed):
            selected.add(check.check_id)
        if "evol_x" in check.entrypoint.replace("\\", "/"):
            if any("evol_x" in p for p in changed):
                selected.add(check.check_id)
    return frozenset(selected)
