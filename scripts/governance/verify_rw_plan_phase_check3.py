#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Strict WP-12 phase check 3 over RW-PLAN + FRZ matrix (mechanical; W2-RECOVERY-02)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
REQUIRED_AUDIT_FLAGS = {"A4-005", "A4-018", "A6-005", "A4-006", "A3-018", "A4-016"}
SCOPE_DEPS = {"T-AF2-RW-PLAN", "T-AF2-RW-PLAN-AUDIT", "PT-V04", "PT-B01", "PT-B02", "REV-MASTER"}


def _canonical_code_fix_ids() -> set[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_rw_plan",
        _REPO / "scripts/governance/build_rw_plan.py",
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load build_rw_plan")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    parse = getattr(mod, "parse_appendix_a_code_fix_inventory")
    return {r.frz_id for r in parse()}


def _lane_pt_indices(lane_order: list) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, entry in enumerate(lane_order):
        if isinstance(entry, (list, tuple)) and len(entry) >= 3 and isinstance(entry[2], str):
            if entry[2].startswith("PT-"):
                idx[entry[2]] = i
    return idx


def validate_plan_structure(data: dict[str, object], canonical: set[str]) -> list[str]:
    errors: list[str] = []
    inventory = data.get("code_fix_inventory")
    if not isinstance(inventory, list) or not inventory:
        errors.append("code_fix_inventory empty or missing")
        return errors

    plan_ids: list[str] = []
    seen: set[str] = set()
    for row in inventory:
        if not isinstance(row, dict):
            errors.append("inventory row not an object")
            continue
        frz = row.get("frz_id")
        if not frz:
            errors.append("row without frz_id")
            continue
        plan_ids.append(str(frz))
        if frz in seen:
            errors.append(f"duplicate frz_id {frz}")
        seen.add(str(frz))
        if not row.get("owner_ticket"):
            errors.append(f"{frz}: missing owner_ticket")
        nt = row.get("negative_test_id")
        if not nt and "gate extension" not in str(row.get("remediation_track", "")):
            errors.append(f"{frz}: missing negative_test_id (strict)")
        if row.get("regression_scope") in (None, "", "MAPPING_MISSING"):
            errors.append(f"{frz}: missing regression_scope mapping")
        status = row.get("red_first_status")
        if status not in ("EVIDENCE_MISSING", "RED_AT_PREFIX", "GREEN_AT_PREFIX"):
            errors.append(f"{frz}: invalid red_first_status {status!r}")

    plan_set = set(plan_ids)
    missing = sorted(canonical - plan_set)
    extra = sorted(plan_set - canonical)
    if missing:
        errors.append(f"missing canonical CODE_FIX rows: {missing}")
    if extra:
        errors.append(f"unexpected CODE_FIX rows in plan: {extra}")

    conflicts = data.get("audit_flag_conflicts") or []
    by_flag: dict[str, int] = {}
    found_flags: set[str] = set()
    for c in conflicts:
        if not isinstance(c, dict):
            continue
        fid = c.get("finding_id")
        if fid:
            by_flag[str(fid)] = by_flag.get(str(fid), 0) + 1
            found_flags.add(str(fid))
            if not c.get("owner_ticket"):
                errors.append(f"audit conflict {fid}: no owner_ticket")
    for dup, count in by_flag.items():
        if count > 1:
            errors.append(f"duplicate audit conflict {dup}")
    if REQUIRED_AUDIT_FLAGS - found_flags:
        errors.append(f"audit_flag_conflicts missing: {sorted(REQUIRED_AUDIT_FLAGS - found_flags)}")

    deps = data.get("dependencies") or {}
    for key in ("W2", "SCOPE", "W3"):
        if key not in deps:
            errors.append(f"dependencies missing {key}")
    scope = deps.get("SCOPE")
    if isinstance(scope, list):
        missing_scope = sorted(SCOPE_DEPS - set(scope))
        if missing_scope:
            errors.append(f"SCOPE missing predecessors: {missing_scope}")

    pt_idx = _lane_pt_indices(data.get("lane_order") or [])
    if "PT-WA" in pt_idx and "PT-WB" in pt_idx and pt_idx["PT-WA"] > pt_idx["PT-WB"]:
        errors.append("lane_order: PT-WB precedes PT-WA (reversed)")
    if not data.get("serialization_pairs"):
        errors.append("serialization_pairs empty")
    if not data.get("overlap_hotspots"):
        errors.append("overlap_hotspots empty")
    return errors


def validate_red_first_evidence(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for row in data.get("code_fix_inventory") or []:
        if not isinstance(row, dict):
            continue
        frz = row.get("frz_id")
        status = row.get("red_first_status")
        sha = row.get("red_first_prefix_sha")
        ref = row.get("red_first_evidence_ref")
        if status == "EVIDENCE_MISSING" and sha:
            errors.append(f"{frz}: EVIDENCE_MISSING must not carry prefix SHA")
        if status == "GREEN_AT_PREFIX":
            errors.append(f"{frz}: GREEN_AT_PREFIX does not demonstrate red-first replay")
        if status == "RED_AT_PREFIX":
            if not sha or not isinstance(sha, str) or len(sha) != 40:
                errors.append(f"{frz}: RED_AT_PREFIX without valid historical prefix SHA")
            if not ref:
                errors.append(f"{frz}: RED_AT_PREFIX without evidence reference")
    return errors


def validate_rw_plan_audit(data: dict[str, object]) -> list[str]:
    notes: list[str] = []
    inventory = data.get("code_fix_inventory") or []
    missing_evidence = sum(
        1 for r in inventory if isinstance(r, dict) and r.get("red_first_status") == "EVIDENCE_MISSING"
    )
    if missing_evidence:
        notes.append(f"historical red-first evidence incomplete for {missing_evidence} CODE_FIX rows")
    notes.append("independent RW-PLAN acceptance not established by mechanical validator")
    return notes


def _emit(plan_errors: list[str], red_first: list[str], audit_notes: list[str], plan_status: str) -> None:
    for e in plan_errors:
        print(f"[PLAN_STRUCTURE] {e}", file=sys.stderr)
    print(f"PLAN_STRUCTURE={plan_status}")
    for e in red_first:
        print(f"[RED_FIRST_EVIDENCE] {e}")
    print("RED_FIRST_EVIDENCE=INCOMPLETE")
    for e in audit_notes:
        print(f"[RW_PLAN_AUDIT] {e}")
    print("RW_PLAN_AUDIT=INCOMPLETE")
    print(f"CHECK3_STRUCTURE_GREEN={'YES' if plan_status == 'PASS' else 'NO'}")


def main() -> int:
    plan_errors: list[str] = []
    if not _RW_JSON.is_file():
        plan_errors.append(f"missing {_RW_JSON.relative_to(_REPO)}")
        _emit(plan_errors, [], ["audit blocked: missing rw plan"], "FAIL")
        return 1

    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    try:
        canonical = _canonical_code_fix_ids()
    except SystemExit as exc:
        plan_errors.append(f"matrix parse failed: {exc}")
        _emit(plan_errors, [], ["audit blocked: matrix parse failed"], "FAIL")
        return 1

    plan_errors.extend(validate_plan_structure(data, canonical))
    red_first_errors = validate_red_first_evidence(data)
    audit_notes = validate_rw_plan_audit(data)
    plan_status = "PASS" if not plan_errors else "FAIL"
    _emit(plan_errors, red_first_errors + audit_notes, audit_notes, plan_status)
    return 0 if plan_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
