# © Artur Czarnecki. All rights reserved.

"""CHANGED-SINCE SSOT path → check_id impact map (C024/C029/C072)."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import ManifestCheck

_IMPACT_RESOURCE = "changed_since_impact_map.v1.json"

_FAIL_CLOSED_PREFIXES = (
    "intergrax/platform/enterprise_audit/",
    "scripts/enterprise_audit/",
)


def fail_closed_prefixes() -> tuple[str, ...]:
    return _FAIL_CLOSED_PREFIXES


def load_impact_map_from_path(path: Path) -> dict[str, frozenset[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    prefixes: dict[str, frozenset[str]] = {}
    for prefix, ids in raw.get("path_prefix_to_check_ids", {}).items():
        normalized = str(prefix).replace("\\", "/").rstrip("/")
        prefixes[normalized] = frozenset(str(i) for i in ids)
    return prefixes


def load_default_impact_map() -> dict[str, frozenset[str]]:
    pkg = resources.files("intergrax.platform.enterprise_audit")
    text = (pkg / "data" / _IMPACT_RESOURCE).read_text(encoding="utf-8")
    raw = json.loads(text)
    prefixes: dict[str, frozenset[str]] = {}
    for prefix, ids in raw.get("path_prefix_to_check_ids", {}).items():
        normalized = str(prefix).replace("\\", "/").rstrip("/")
        prefixes[normalized] = frozenset(str(i) for i in ids)
    return prefixes


def build_impact_map_from_manifest(checks: tuple[ManifestCheck, ...]) -> dict[str, frozenset[str]]:
    """Derive longest-prefix owners from manifest entrypoints + platform module paths."""
    mapping: dict[str, set[str]] = {}

    def add(prefix: str, check_id: str) -> None:
        key = prefix.replace("\\", "/").rstrip("/")
        mapping.setdefault(key, set()).add(check_id)

    for check in checks:
        ep = check.entrypoint.replace("\\", "/")
        if "::" in ep:
            ep = ep.split("::", 1)[0]
        add(ep, check.check_id)
        parts = Path(ep).parts
        if len(parts) > 1:
            add("/".join(parts[:-1]), check.check_id)

    platform_rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("intergrax/platform/enterprise_audit/changed_since.py", ("C024", "C025", "C029", "C072")),
        ("intergrax/platform/enterprise_audit/impact_map.py", ("C024", "C029", "C072")),
        (
            "intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json",
            ("C024", "C029"),
        ),
        ("intergrax/platform/enterprise_audit/sandbox.py", ("C014", "C015", "C016", "C017")),
        ("intergrax/platform/enterprise_audit/report.py", ("C013",)),
        ("intergrax/platform/enterprise_audit/baseline.py", ("C022", "C023", "C028", "C026")),
        ("intergrax/platform/enterprise_audit/runner.py", ("C004", "C002", "C009")),
        ("intergrax/platform/enterprise_audit/modes.py", ("C011", "C067", "C071")),
        ("intergrax/platform/enterprise_audit/validators.py", ("C026", "C027", "C069")),
        ("intergrax/platform/enterprise_audit/evidence.py", ("C006", "C068", "C073", "C049")),
        ("intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json", ("C019", "C021", "C030")),
        ("scripts/enterprise_audit/build_impact_map.py", ("C029", "C024")),
        ("scripts/enterprise_audit/run_qual_wave.py", ("C051", "C074")),
    )
    for path, ids in platform_rules:
        for cid in ids:
            add(path, cid)

    return {k: frozenset(v) for k, v in mapping.items()}


def check_ids_for_changed_path(path: str, prefixes: dict[str, frozenset[str]]) -> frozenset[str]:
    normalized = path.replace("\\", "/")
    matched: set[str] = set()
    for prefix, ids in sorted(prefixes.items(), key=lambda kv: len(kv[0]), reverse=True):
        if normalized == prefix or normalized.startswith(prefix + "/"):
            matched.update(ids)
            break
    return frozenset(matched)


def unmapped_fail_closed_paths(
    changed: frozenset[str],
    *,
    prefixes: dict[str, frozenset[str]],
    entrypoint_covered: frozenset[str],
) -> frozenset[str]:
    unmapped: set[str] = set()
    for path in changed:
        if not any(path.startswith(p) for p in _FAIL_CLOSED_PREFIXES):
            continue
        if path in entrypoint_covered:
            continue
        if check_ids_for_changed_path(path, prefixes):
            continue
        unmapped.add(path)
    return frozenset(unmapped)


def selected_check_ids_from_impact(
    changed: frozenset[str],
    *,
    prefixes: dict[str, frozenset[str]],
) -> frozenset[str]:
    selected: set[str] = set()
    for path in changed:
        selected.update(check_ids_for_changed_path(path, prefixes))
    return frozenset(selected)
