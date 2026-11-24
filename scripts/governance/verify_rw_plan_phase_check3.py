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
CANONICAL_SCOPE_PREDECESSORS = frozenset(
    {
        "M1",
        "DS-1",
        "DS-2",
        "DS-3",
        "DS-4",
        "DS-EXC",
        "PT-V04",
        "PT-B01",
        "PT-B02",
        "REV-MASTER",
        "T-AF2-RW-PLAN",
        "T-AF2-RW-PLAN-AUDIT",
    }
)

A4_006_AUDIT_CONTRACT: dict[str, object] = {
    "owner_ticket": None,
    "owner_node": "PT-T01",
    "owner_wave": "W3",
    "owner_source": "WP12 DAG / PT-T01 / A4-006",
}

EXPLICIT_OWNERSHIP_CONTRACT_KEYS = (
    "owner_ticket",
    "owner_node",
    "owner_wave",
    "owner_source",
    "verification_node",
    "verification_wave",
)


def _load_build_rw_plan():
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
    return mod


def _canonical_code_fix_ids() -> set[str]:
    mod = _load_build_rw_plan()
    parse = getattr(mod, "parse_appendix_a_code_fix_inventory")
    return {r.frz_id for r in parse()}


def _matrix_owners_by_frz() -> dict[str, str | None]:
    mod = _load_build_rw_plan()
    parse = getattr(mod, "parse_appendix_a_code_fix_inventory")
    return {r.frz_id: r.owner_ticket for r in parse()}


def _explicit_ownership_contracts() -> dict[str, dict[str, object]]:
    mod = _load_build_rw_plan()
    return dict(mod.EXPLICIT_CODE_FIX_OWNERSHIP)


def _explicit_contract_satisfied(row: dict, contract: dict[str, object]) -> bool:
    if row.get("owner_provenance") != "explicit_mep_wp12":
        return False
    for key in EXPLICIT_OWNERSHIP_CONTRACT_KEYS:
        if key not in contract:
            continue
        if row.get(key) != contract[key]:
            return False
    if contract.get("owner_ticket") is None and row.get("owner_ticket"):
        return False
    return True


def _ownership_verified(
    row: dict, matrix_owners: dict[str, str | None], explicit: dict[str, dict[str, object]]
) -> bool:
    frz = str(row.get("frz_id", ""))
    contract = explicit.get(frz)
    if contract is not None:
        return _explicit_contract_satisfied(row, contract)

    node = row.get("owner_node")
    if node:
        return False
    ticket = row.get("owner_ticket")
    expected = matrix_owners.get(frz)
    if expected is not None:
        return ticket == expected
    return bool(ticket)


def _lane_pt_indices(lane_order: list) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, entry in enumerate(lane_order):
        if isinstance(entry, (list, tuple)) and len(entry) >= 3 and isinstance(entry[2], str):
            if entry[2].startswith("PT-"):
                idx[entry[2]] = i
    return idx


def _validate_scope_dependencies(deps: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for key in ("W2", "SCOPE", "W3"):
        if key not in deps:
            errors.append(f"dependencies missing {key}")
    scope = deps.get("SCOPE")
    if scope is None:
        errors.append("SCOPE dependency collection missing")
        return errors
    if not isinstance(scope, list):
        errors.append("SCOPE dependency value must be a list")
        return errors
    if not scope:
        errors.append("SCOPE dependency collection empty")
        return errors
    if len(scope) != len(set(scope)):
        errors.append("SCOPE duplicate predecessors")
    scope_set = set(scope)
    missing_scope = sorted(CANONICAL_SCOPE_PREDECESSORS - scope_set)
    if missing_scope:
        errors.append(f"SCOPE missing predecessors: {missing_scope}")
    extra = sorted(scope_set - CANONICAL_SCOPE_PREDECESSORS)
    if extra:
        errors.append(f"SCOPE unexpected predecessors: {extra}")
    return errors


def _validate_lane_order(lane_order: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(lane_order, list):
        errors.append("lane_order must be a list")
        return errors
    pt_labels = [
        entry[2]
        for entry in lane_order
        if isinstance(entry, (list, tuple))
        and len(entry) >= 3
        and isinstance(entry[2], str)
        and entry[2].startswith("PT-")
    ]
    wa_count = pt_labels.count("PT-WA")
    wb_count = pt_labels.count("PT-WB")
    if wa_count == 0:
        errors.append("lane_order: missing PT-WA lane")
    if wb_count == 0:
        errors.append("lane_order: missing PT-WB lane")
    if wa_count > 1:
        errors.append("lane_order: duplicate PT-WA lane")
    if wb_count > 1:
        errors.append("lane_order: duplicate PT-WB lane")
    pt_idx = _lane_pt_indices(lane_order)
    if "PT-WA" in pt_idx and "PT-WB" in pt_idx and pt_idx["PT-WA"] > pt_idx["PT-WB"]:
        errors.append("lane_order: PT-WB precedes PT-WA (reversed)")
    return errors


def _validate_audit_conflict(conflict: dict) -> list[str]:
    errors: list[str] = []
    fid = str(conflict.get("finding_id", ""))
    if fid == "A4-006":
        for key, expected in A4_006_AUDIT_CONTRACT.items():
            if conflict.get(key) != expected:
                errors.append(f"audit conflict {fid}: canonical PT-T01 contract mismatch on {key}")
        if conflict.get("owner_ticket") == "T-IGX-044-A":
            errors.append(f"audit conflict {fid}: T-IGX-044-A is not PT-T01 ownership")
        return errors
    if not conflict.get("owner_ticket"):
        errors.append(f"audit conflict {fid}: no owner_ticket")
    return errors


def validate_plan_structure(
    data: dict[str, object],
    canonical: set[str],
    *,
    matrix_owners: dict[str, str | None] | None = None,
    explicit: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    errors: list[str] = []
    if matrix_owners is None:
        matrix_owners = _matrix_owners_by_frz()
    if explicit is None:
        explicit = _explicit_ownership_contracts()

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
        if row.get("owner_ticket") and row.get("owner_node") and str(frz) not in explicit:
            if not row.get("owner_provenance"):
                errors.append(f"{frz}: contradictory ticket and node attribution without provenance")
        if frz in explicit and not row.get("owner_provenance"):
            errors.append(f"{frz}: missing required provenance for canonical ownership")
        if not _ownership_verified(row, matrix_owners, explicit):
            errors.append(f"{frz}: ownership not verified (missing ticket or canonical node contract)")
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
            errors.extend(_validate_audit_conflict(c))
    for dup, count in by_flag.items():
        if count > 1:
            errors.append(f"duplicate audit conflict {dup}")
    if REQUIRED_AUDIT_FLAGS - found_flags:
        errors.append(f"audit_flag_conflicts missing: {sorted(REQUIRED_AUDIT_FLAGS - found_flags)}")

    deps = data.get("dependencies") or {}
    errors.extend(_validate_scope_dependencies(deps))
    errors.extend(_validate_lane_order(data.get("lane_order") or []))
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
    return ["independent RW-PLAN acceptance not performed by mechanical validator"]


def count_verified_ownership(data: dict[str, object]) -> tuple[int, int]:
    matrix_owners = _matrix_owners_by_frz()
    explicit = _explicit_ownership_contracts()
    inventory = data.get("code_fix_inventory") or []
    verified = sum(1 for r in inventory if isinstance(r, dict) and _ownership_verified(r, matrix_owners, explicit))
    return verified, len(inventory)


def _emit(
    plan_errors: list[str],
    red_first: list[str],
    audit_notes: list[str],
    plan_status: str,
    *,
    ownership_verified: int,
    ownership_total: int,
) -> None:
    for e in plan_errors:
        print(f"[PLAN_STRUCTURE] {e}", file=sys.stderr)
    print(f"PLAN_STRUCTURE={plan_status}")
    print(f"CODE_FIX_OWNERSHIP={ownership_verified}/{ownership_total}")
    for e in red_first:
        print(f"[RED_FIRST_EVIDENCE] {e}")
    if red_first:
        print("RED_FIRST_EVIDENCE=INCOMPLETE")
    else:
        print("RED_FIRST_EVIDENCE=INCOMPLETE")
    for e in audit_notes:
        print(f"[RW_PLAN_AUDIT] {e}")
    print("RW_PLAN_AUDIT=NOT_PERFORMED")
    print(f"CHECK3_STRUCTURE_GREEN={'YES' if plan_status == 'PASS' else 'NO'}")


def main() -> int:
    plan_errors: list[str] = []
    if not _RW_JSON.is_file():
        plan_errors.append(f"missing {_RW_JSON.relative_to(_REPO)}")
        _emit(plan_errors, [], ["audit blocked: missing rw plan"], "FAIL", ownership_verified=0, ownership_total=0)
        return 1

    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    try:
        canonical = _canonical_code_fix_ids()
    except SystemExit as exc:
        plan_errors.append(f"matrix parse failed: {exc}")
        _emit(plan_errors, [], ["audit blocked: matrix parse failed"], "FAIL", ownership_verified=0, ownership_total=0)
        return 1

    ownership_verified, ownership_total = count_verified_ownership(data)
    plan_errors.extend(validate_plan_structure(data, canonical))
    red_first_errors = validate_red_first_evidence(data)
    audit_notes = validate_rw_plan_audit(data)
    plan_status = "PASS" if not plan_errors else "FAIL"
    _emit(
        plan_errors,
        red_first_errors,
        audit_notes,
        plan_status,
        ownership_verified=ownership_verified,
        ownership_total=ownership_total,
    )
    return 0 if plan_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
