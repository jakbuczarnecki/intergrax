# © Artur Czarnecki. All rights reserved.

"""Load and validate closed-world enterprise audit check manifest."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditMode,
    AuditSeverity,
    ManifestCheck,
)

_MANIFEST_RESOURCE = "enterprise_audit_check_manifest.v1.json"


def _parse_modes(raw: list[str]) -> frozenset[AuditMode]:
    return frozenset(AuditMode(m) for m in raw)


def _parse_check(raw: dict) -> ManifestCheck:
    return ManifestCheck(
        check_id=raw["check_id"],
        entrypoint_kind=raw["entrypoint_kind"],
        entrypoint=raw["entrypoint"],
        mutating=bool(raw.get("mutating", False)),
        modes=_parse_modes(raw["modes"]),
        severity_default=AuditSeverity(raw.get("severity_default", "ERROR")),
        category=AuditCategory(raw.get("category", "architecture")),
        pg_id=str(raw.get("pg_id", "")),
        depends_on=frozenset(raw.get("depends_on", ())),
        fast_mandatory=bool(raw.get("fast_mandatory", False)),
        schema_version=int(raw.get("schema_version", 1)),
    )


def load_manifest_from_path(path: Path) -> tuple[ManifestCheck, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    checks = tuple(_parse_check(item) for item in data["checks"])
    _validate_manifest(checks)
    return checks


def load_default_manifest() -> tuple[ManifestCheck, ...]:
    pkg = resources.files("intergrax.platform.enterprise_audit")
    text = (pkg / "data" / _MANIFEST_RESOURCE).read_text(encoding="utf-8")
    data = json.loads(text)
    checks = tuple(_parse_check(item) for item in data["checks"])
    _validate_manifest(checks)
    return checks


def _validate_manifest(checks: tuple[ManifestCheck, ...]) -> None:
    ids = [c.check_id for c in checks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate check_id in enterprise audit manifest")
    if len(checks) != 85:
        raise ValueError(f"enterprise audit manifest must contain 85 checks, got {len(checks)}")


def checks_for_mode(
    checks: tuple[ManifestCheck, ...],
    mode: AuditMode,
    *,
    changed_check_ids: frozenset[str] | None = None,
) -> tuple[ManifestCheck, ...]:
    if mode is AuditMode.CHANGED_SINCE:
        if not changed_check_ids:
            return ()
        return tuple(c for c in checks if c.check_id in changed_check_ids)
    selected: list[ManifestCheck] = []
    for check in checks:
        if mode not in check.modes:
            continue
        if mode is AuditMode.FAST and check.entrypoint_kind == "internal_pending":
            continue
        selected.append(check)
    return tuple(selected)
