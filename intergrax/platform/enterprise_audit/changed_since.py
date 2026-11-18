# © Artur Czarnecki. All rights reserved.

"""CHANGED-SINCE check selection via git diff + invariant floor."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import AuditMode, ManifestCheck
from intergrax.platform.enterprise_audit.impact_map import (
    check_ids_for_changed_path,
    is_product_path,
    load_default_impact_map,
    selected_check_ids_from_impact,
    unmapped_fail_closed_paths,
    unmapped_product_paths,
)
from intergrax.platform.enterprise_audit.manifest import checks_for_mode
from intergrax.platform.enterprise_audit.repo_identity import assert_git_ancestor

_REASON_SAMPLE_SIZE = 8


def git_changed_files(repo_root: Path, since_sha: str) -> frozenset[str]:
    root = repo_root.resolve()
    completed = subprocess.run(
        # --no-renames: both sides of a rename are listed, so a move out of an unmapped path stays visible.
        ["git", "diff", "--name-only", "--no-renames", f"{since_sha}..HEAD"],
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
    *,
    prefixes: dict[str, frozenset[str]] | None = None,
) -> frozenset[str]:
    """Paths under evol_x/ changed but not mapped to any manifest check (fail-closed).

    Without ``prefixes`` only entrypoint files count as mapped (legacy behaviour). With ``prefixes`` a path is also
    mapped when its longest impact-map prefix names at least one manifest check id.
    """
    mapping = evol_checks_by_path(checks)
    manifest_ids = frozenset(c.check_id for c in checks)
    unmapped: set[str] = set()
    for path in changed:
        if not path.startswith(_EVOL_PREFIX):
            continue
        matched = False
        for ep, _ids in mapping.items():
            if path == ep or path.startswith(ep.rstrip("/") + "/"):
                matched = True
                break
        if not matched and prefixes is not None:
            matched = bool(check_ids_for_changed_path(path, prefixes) & manifest_ids)
        if not matched:
            unmapped.add(path)
    return frozenset(unmapped)


@dataclass(frozen=True)
class ChangedSinceSelection:
    """Outcome of CHANGED-SINCE planning: the selected ids and, when the selection escalated, why."""

    check_ids: frozenset[str]
    escalated_to_full: bool
    unmapped_paths: tuple[str, ...]
    reasons: tuple[str, ...]


def plan_changed_since_selection(
    checks: tuple[ManifestCheck, ...],
    *,
    repo_root: Path,
    since_sha: str,
) -> ChangedSinceSelection:
    """Plan CHANGED-SINCE selection; an unmapped changed product path escalates to the FULL check set (DP-66)."""
    assert_git_ancestor(repo_root, ancestor_sha=since_sha)
    changed = frozenset(path.replace("\\", "/") for path in git_changed_files(repo_root, since_sha))
    impact_prefixes = load_default_impact_map()
    unmapped = unmapped_evol_changed_paths(checks, changed, prefixes=impact_prefixes)
    if unmapped:
        raise ValueError("CHANGED-SINCE fail-closed: unmapped evol_x paths: " + ", ".join(sorted(unmapped)[:8]))
    manifest_ids = frozenset(c.check_id for c in checks)
    entrypoint_covered = frozenset(
        path for path in changed if any(_check_touched(check, frozenset({path})) for check in checks)
    )
    # Fail-closed prefixes that are not product paths (guard; the product tree is handled by default-deny below).
    platform_unmapped = unmapped_fail_closed_paths(
        frozenset(path for path in changed if not is_product_path(path)),
        prefixes=impact_prefixes,
        entrypoint_covered=entrypoint_covered,
    )
    if platform_unmapped:
        sample = ", ".join(sorted(platform_unmapped)[:_REASON_SAMPLE_SIZE])
        raise ValueError(f"CHANGED-SINCE fail-closed: unmapped platform paths: {sample}")
    unmapped_product = unmapped_product_paths(
        changed,
        prefixes=impact_prefixes,
        manifest_check_ids=manifest_ids,
        entrypoint_covered=entrypoint_covered,
    )
    if unmapped_product:
        ordered = tuple(sorted(unmapped_product))
        full_ids = frozenset(c.check_id for c in checks_for_mode(checks, AuditMode.FULL))
        sample = ", ".join(ordered[:_REASON_SAMPLE_SIZE])
        reason = (
            f"CHANGED-SINCE escalated to FULL: {len(ordered)} changed product path(s) map to no manifest check "
            f"(first {min(len(ordered), _REASON_SAMPLE_SIZE)}: {sample})"
        )
        return ChangedSinceSelection(
            check_ids=full_ids,
            escalated_to_full=True,
            unmapped_paths=ordered,
            reasons=(reason,),
        )
    selected: set[str] = set(invariant_floor(checks))
    selected.update(selected_check_ids_from_impact(changed, prefixes=impact_prefixes) & manifest_ids)
    for check in checks:
        if _check_touched(check, changed):
            selected.add(check.check_id)
        if "evol_x" in check.entrypoint.replace("\\", "/"):
            if any("evol_x" in p for p in changed):
                selected.add(check.check_id)
    return ChangedSinceSelection(
        check_ids=frozenset(selected),
        escalated_to_full=False,
        unmapped_paths=(),
        reasons=(),
    )


def select_checks_for_changed_since(
    checks: tuple[ManifestCheck, ...],
    *,
    repo_root: Path,
    since_sha: str,
) -> frozenset[str]:
    return plan_changed_since_selection(checks, repo_root=repo_root, since_sha=since_sha).check_ids
