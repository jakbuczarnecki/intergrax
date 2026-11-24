#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Strict WP-12 phase check 3 over RW-PLAN + FRZ matrix (mechanical; W2-RECOVERY-01)."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
_RW_MD = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/RW_PLAN.md"

REQUIRED_AUDIT_FLAGS = {"A4-005", "A4-018", "A6-005", "A4-006", "A3-018", "A4-016"}


def main() -> int:
    errors: list[str] = []
    if not _RW_MD.is_file():
        errors.append(f"missing {_RW_MD.relative_to(_REPO)}")
    if not _RW_JSON.is_file():
        errors.append(f"missing {_RW_JSON.relative_to(_REPO)}")
        if errors:
            for e in errors:
                print(f"[FAIL] {e}")
            return 1

    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    inventory = data.get("code_fix_inventory")
    if not isinstance(inventory, list) or not inventory:
        errors.append("code_fix_inventory empty or missing")

    frz_seen: set[str] = set()
    for row in inventory or []:
        frz = row.get("frz_id")
        if not frz:
            errors.append("row without frz_id")
            continue
        if frz in frz_seen:
            errors.append(f"duplicate frz_id {frz}")
        frz_seen.add(frz)
        owner = row.get("owner_ticket")
        if not owner:
            errors.append(f"{frz}: missing owner_ticket")
        nt = row.get("negative_test_id")
        if not nt and "gate extension" not in str(row.get("remediation_track", "")):
            errors.append(f"{frz}: missing negative_test_id (strict)")
        if nt in ("NEG-0", "NEG-P"):
            pass  # matrix criterion types; no red-first NT id required
        status = row.get("red_first_status")
        if status not in ("EVIDENCE_MISSING", "RED_AT_PREFIX", "GREEN_AT_PREFIX"):
            errors.append(f"{frz}: invalid red_first_status {status!r}")
        if not row.get("regression_scope"):
            errors.append(f"{frz}: missing regression_scope")

    conflicts = data.get("audit_flag_conflicts") or []
    found = {c.get("finding_id") for c in conflicts if isinstance(c, dict)}
    missing_flags = REQUIRED_AUDIT_FLAGS - found
    if missing_flags:
        errors.append(f"audit_flag_conflicts missing: {sorted(missing_flags)}")
    for c in conflicts:
        if not c.get("owner_ticket"):
            errors.append(f"audit conflict {c.get('finding_id')}: no owner_ticket")

    deps = data.get("dependencies") or {}
    for key in ("W2", "SCOPE", "W3"):
        if key not in deps:
            errors.append(f"dependencies missing {key}")

    if not data.get("serialization_pairs"):
        errors.append("serialization_pairs empty")
    if not data.get("overlap_hotspots"):
        errors.append("overlap_hotspots empty")

    if errors:
        print("[FAIL] verify_rw_plan_phase_check3:")
        for e in errors:
            print(f"  - {e}")
        print(f"CHECK3_GREEN=NO rows={len(inventory or [])}")
        return 1

    print("[OK] verify_rw_plan_phase_check3: strict mechanical checks passed")
    print(f"CHECK3_GREEN=YES rows={len(inventory)} audit_flags={len(conflicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
