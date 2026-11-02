# © Artur Czarnecki. All rights reserved.

"""Build enterprise_audit_check_manifest.v1.json from Annex G2 (85 rows)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
G2 = ROOT / "docs/project/maintainers/qualification/ent_audit_x/ANNEX_G2_CONTROL_GATE_MAPPING.md"
OUT = ROOT / "intergrax/platform/enterprise_audit/data/enterprise_audit_check_manifest.v1.json"

ROW = re.compile(
    r"^\|\s*(?P<id>[^|]+?)\s*\|\s*(?P<pg>[^|]+?)\s*\|\s*`(?P<ep>[^`]+)`",
)


def _category(check_id: str) -> str:
    if check_id.startswith("TEN-"):
        return "tenant"
    if check_id.startswith("VBN-"):
        return "vbn"
    return "architecture"


def _entrypoint_kind(entrypoint: str) -> str:
    if entrypoint.startswith("TO_IMPLEMENT"):
        return "internal_pending"
    if "(allowlisted" in entrypoint:
        return "internal_pending"
    if entrypoint.endswith(".py") and "::" not in entrypoint and "/" in entrypoint:
        if entrypoint.startswith("scripts/"):
            return "python_script"
        return "pytest_file"
    if "::" in entrypoint:
        return "pytest_nodeid"
    if entrypoint.startswith("scripts/"):
        return "python_script"
    return "internal_pending"


def _modes(kind: str) -> list[str]:
    if kind == "internal_pending":
        return ["FULL", "FREEZE", "CHANGED-SINCE"]
    return ["FAST", "FULL", "FREEZE", "CHANGED-SINCE"]


def main() -> None:
    checks: list[dict] = []
    for line in G2.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if not match:
            continue
        check_id = match.group("id").strip()
        if not check_id.startswith(("ENT-", "TEN-", "VBN-")):
            continue
        entrypoint = match.group("ep").strip()
        pg = match.group("pg").strip()
        kind = _entrypoint_kind(entrypoint)
        checks.append(
            {
                "check_id": check_id,
                "pg_id": pg,
                "entrypoint_kind": kind,
                "entrypoint": entrypoint,
                "mutating": False,
                "modes": _modes(kind),
                "severity_default": "ERROR",
                "category": _category(check_id),
            }
        )
    if len(checks) != 85:
        raise SystemExit(f"expected 85 checks, parsed {len(checks)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"version": 1, "checks": checks}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(checks)} checks)")


if __name__ == "__main__":
    main()
