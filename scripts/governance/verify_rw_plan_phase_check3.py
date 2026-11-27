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
    "owner_ticket": "T-IGX-078-D",
    "owner_node": "PT-T01",
    "owner_wave": "W3",
    "owner_source": "WP12 DAG / PT-T01 / A4-006; GA.2R T-IGX-078-D",
}

_GA2R_RECORD_NAME = "RED_FIRST_GA2R_RESULT"

EXPLICIT_OWNERSHIP_CONTRACT_KEYS = (
    "owner_ticket",
    "owner_node",
    "owner_wave",
    "owner_source",
    "verification_node",
    "verification_wave",
    "owner_ticket_family",
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


def _matrix_row_has_unauthorized_explicit_metadata(row: dict) -> bool:
    if row.get("owner_provenance"):
        return True
    for key in EXPLICIT_OWNERSHIP_CONTRACT_KEYS:
        if key == "owner_ticket":
            continue
        if row.get(key) is not None:
            return True
    return False


def _explicit_contract_satisfied(row: dict, contract: dict[str, object]) -> bool:
    if row.get("owner_provenance") != contract.get("owner_provenance"):
        return False
    if "owner_ticket" not in row:
        return False
    for key in EXPLICIT_OWNERSHIP_CONTRACT_KEYS:
        if key in contract:
            if row.get(key) != contract[key]:
                return False
        elif row.get(key) is not None:
            return False
    return True


def _ownership_verified(
    row: dict, matrix_owners: dict[str, str | None], explicit: dict[str, dict[str, object]]
) -> bool:
    frz = str(row.get("frz_id", ""))
    contract = explicit.get(frz)
    if contract is not None:
        return _explicit_contract_satisfied(row, contract)

    if _matrix_row_has_unauthorized_explicit_metadata(row):
        return False
    if row.get("owner_node"):
        return False
    ticket = row.get("owner_ticket")
    expected = matrix_owners.get(frz)
    return isinstance(expected, str) and bool(expected) and isinstance(ticket, str) and ticket == expected


def _lane_pt_indices(lane_order: list) -> dict[str, int]:
    idx: dict[str, int] = {}
    for i, entry in enumerate(lane_order):
        if isinstance(entry, (list, tuple)) and len(entry) >= 3 and isinstance(entry[2], str):
            if entry[2].startswith("PT-"):
                idx[entry[2]] = i
    return idx


def _validate_scope_dependencies(deps: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(deps, dict):
        errors.append("dependencies container must be a dictionary")
        return errors
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
    for i, entry in enumerate(scope):
        if not isinstance(entry, str):
            errors.append(f"SCOPE dependency entry {i} must be a nonempty string")
            continue
        if not entry:
            errors.append(f"SCOPE dependency entry {i} must be a nonempty string")
            continue
        if entry != entry.strip():
            errors.append(f"SCOPE dependency entry {i} has leading or trailing whitespace")
    if any(not isinstance(e, str) or not e or e != e.strip() for e in scope):
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


def _is_identifier(value: object, markers: tuple[str, ...]) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value not in markers and "gate extension" not in value


def _negative_test_errors(row: dict, build) -> list[str]:
    errors: list[str] = []
    frz = row.get("frz_id")
    if not _is_identifier(row.get("negative_test_id"), build.CLASS_MARKERS):
        errors.append(f"{frz}: negative_test_id is missing or is a class marker (strict)")
    entries = row.get("negative_tests")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{frz}: negative_tests must name at least one node")
        return errors
    required = (*build.NEGATIVE_TEST_ENTRY_KEYS, "file_state_at_pin")
    for entry in entries:
        missing = [key for key in required if not (isinstance(entry, dict) and entry.get(key))]
        if missing:
            errors.append(f"{frz}: negative test entry misses {', '.join(missing)}")
        elif not _is_identifier(entry["nt_id"], build.CLASS_MARKERS):
            errors.append(f"{frz}: negative test entry nt_id is a class marker")
    return errors


def count_named_negative_tests(data: dict[str, object]) -> tuple[int, int]:
    build = _load_build_rw_plan()
    inventory = data.get("code_fix_inventory") or []
    named = sum(1 for row in inventory if isinstance(row, dict) and not _negative_test_errors(row, build))
    return named, len(inventory)


_OD_AF_CLOSE_B2 = "OD-AF-CLOSE-B2"
_EVIDENCE_MISSING_OP_B: dict[str, str] = {
    "FRZ-BND-05": "T-IGX-043-H",
    "FRZ-BND-06": "T-IGX-043-J",
    "FRZ-DEBT-02": "T-IGX-051-H",
    "FRZ-DEBT-03": "T-IGX-051-H",
    "FRZ-DEBT-04": "T-IGX-051-F",
    "FRZ-EXE-02": "T-IGX-043-A",
    "FRZ-PLG-05": "T-IGX-043-A",
    "FRZ-SEC-06": "T-IGX-051-B",
    "FRZ-TEN-02": "T-IGX-078-D",
    "FRZ-TYP-01": "T-IGX-078-C",
    "FRZ-TYP-03": "T-IGX-078-C",
    "FRZ-TYP-05": "T-IGX-051-G",
}
_FRZ_FRZ_10_SCHEDULE = {
    "frz_id": "FRZ-FRZ-10",
    "pre_g2_execution_ticket": "T-IGX-052-I",
    "functional_owner_node": "T-B7-PFP-ACTIVATE",
    "functional_owner_wave": "W5",
    "verification_node": "T-B7-PFP-VERIFY",
    "verification_wave": "W8",
}
_SEEDED_RED_REQUIRED = frozenset({"FRZ-DEBT-02", "FRZ-DEBT-03"})


def _validate_evidence_missing_obligations(inventory: list[object]) -> list[str]:
    errors: list[str] = []
    missing_rows: list[str] = []
    for row in inventory:
        if not isinstance(row, dict):
            continue
        frz = str(row.get("frz_id"))
        status = row.get("red_first_status")
        if status == "PASS":
            errors.append(f"{frz}: EVIDENCE_MISSING must not be promoted to PASS")
        if status != "EVIDENCE_MISSING":
            continue
        if frz in _EVIDENCE_MISSING_OP_B:
            missing_rows.append(frz)
            expected_owner = _EVIDENCE_MISSING_OP_B[frz]
            if row.get("owner_ticket") != expected_owner:
                errors.append(f"{frz}: EVIDENCE_MISSING owner must be {expected_owner}")
            if row.get("operator_decision_id") != _OD_AF_CLOSE_B2:
                errors.append(f"{frz}: missing operator_decision_id {_OD_AF_CLOSE_B2}")
            if not str(row.get("operator_decision_ref") or "").strip():
                errors.append(f"{frz}: missing operator_decision_ref")
            if not str(row.get("evidence_negative_test") or "").strip():
                errors.append(f"{frz}: missing planned negative test reference")
            if not str(row.get("evidence_mechanism") or "").strip():
                errors.append(f"{frz}: missing evidence mechanism")
            if not str(row.get("evidence_certification_obligation") or "").strip():
                errors.append(f"{frz}: missing certification obligation before row close")
            if frz in _SEEDED_RED_REQUIRED and not row.get("evidence_seeded_red_required"):
                errors.append(f"{frz}: must keep seeded-red requirement (FRZ-DEBT-02/03)")
    if sorted(missing_rows) != sorted(_EVIDENCE_MISSING_OP_B):
        errors.append(
            "EVIDENCE_MISSING OP-B row set mismatch: "
            f"expected {sorted(_EVIDENCE_MISSING_OP_B)}, got {sorted(missing_rows)}"
        )
    return errors


def _validate_post_freeze_policy_schedule(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    schedule = data.get("post_freeze_policy_schedule")
    if not isinstance(schedule, dict):
        errors.append("post_freeze_policy_schedule missing")
        return errors
    for key, expected in _FRZ_FRZ_10_SCHEDULE.items():
        if schedule.get(key) != expected:
            errors.append(f"post_freeze_policy_schedule.{key}: expected {expected!r}, got {schedule.get(key)!r}")
    tickets = data.get("tickets") or []
    pfp = next((t for t in tickets if isinstance(t, dict) and t.get("ticket") == "T-IGX-052-I"), None)
    if pfp is None:
        errors.append("T-IGX-052-I missing from tickets table")
    else:
        if pfp.get("wave") != "pre-G2":
            errors.append("T-IGX-052-I must use wave pre-G2 (not W3 functional activation)")
        policy = pfp.get("post_freeze_policy")
        if not isinstance(policy, dict):
            errors.append("T-IGX-052-I missing post_freeze_policy block")
        elif policy.get("functional_owner_wave") != "W5" or policy.get("verification_wave") != "W8":
            errors.append("T-IGX-052-I post_freeze_policy must name W5 activation and W8 verification")
    for lane in data.get("lane_order") or []:
        if isinstance(lane, (list, tuple)) and len(lane) >= 2 and "T-IGX-052-I" in lane[1]:
            label = lane[2] if len(lane) > 2 else ""
            if "W3" in str(label) and "pre_g2" not in str(lane[0]).lower():
                errors.append("T-IGX-052-I must not appear in a W3 execution lane")
    frz_row = next(
        (r for r in data.get("code_fix_inventory") or [] if isinstance(r, dict) and r.get("frz_id") == "FRZ-FRZ-10"),
        None,
    )
    if isinstance(frz_row, dict):
        if frz_row.get("pre_g2_execution_ticket") != "T-IGX-052-I":
            errors.append("FRZ-FRZ-10 must name pre_g2_execution_ticket T-IGX-052-I")
        if frz_row.get("owner_wave") != "W5" or frz_row.get("verification_wave") != "W8":
            errors.append("FRZ-FRZ-10 must keep W5/W8 canonical ownership")
        if frz_row.get("red_first_status") != "RED_FIRST_LATER_WAVE":
            errors.append("FRZ-FRZ-10 must stay RED_FIRST_LATER_WAVE")
    return errors


def _train_errors(data: dict[str, object], build) -> list[str]:
    """Re-check overlap matrix, merge order and lanes from the payload's own write sets."""
    tickets = data.get("tickets")
    if not isinstance(tickets, list) or not tickets:
        return ["tickets table empty or missing"]
    write_sets = {t["ticket"]: [p["path"] for p in t["write_set"]] for t in tickets}
    landed = {t["ticket"] for t in tickets if t.get("landed")}
    open_tickets = set(write_sets) - landed
    w3 = {t["ticket"] for t in tickets if t["ticket"] in open_tickets and t.get("wave", "W3") == "W3"}
    edges = [(e["first"], e["second"]) for e in data.get("serialization_edges") or []]
    errors: list[str] = []
    recomputed = build.overlap_matrix(write_sets)
    if data.get("overlap_matrix") != recomputed:
        errors.append("overlap_matrix is stale or missing (does not match the ticket write sets)")
    try:
        build.check_merge_order(set(write_sets), landed, edges, recomputed)
    except SystemExit as exc:
        errors.append(str(exc))
    lanes = [
        {"lane": e[0], "tickets": list(e[1]), "label": e[2]}
        for e in data.get("lane_order") or []
        if isinstance(e, (list, tuple)) and len(e) == 3 and isinstance(e[1], list)
    ]
    try:
        build.check_lanes(lanes, write_sets, edges, open_tickets, w3)
    except SystemExit as exc:
        errors.append(str(exc))
    return errors


def _coverage_errors(data: dict[str, object], open_outside_inventory: set[str]) -> list[str]:
    entries = [c for c in data.get("coverage") or [] if isinstance(c, dict)]
    covered = {str(c.get("frz_id")) for c in entries}
    errors = [f"coverage missing OPEN row {frz}" for frz in sorted(open_outside_inventory - covered)]
    errors.extend(
        f"coverage {c.get('frz_id')} has no reason or owning node"
        for c in entries
        if not c.get("reason") or not c.get("owner_nodes")
    )
    return errors


def validate_plan_structure(
    data: dict[str, object],
    canonical: set[str],
    *,
    matrix_owners: dict[str, str | None] | None = None,
    explicit: dict[str, dict[str, object]] | None = None,
    open_outside_inventory: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if matrix_owners is None:
        matrix_owners = _matrix_owners_by_frz()
    if explicit is None:
        explicit = _explicit_ownership_contracts()
    build = _load_build_rw_plan()

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
        errors.extend(_negative_test_errors(row, build))
        trace = row.get("traceability")
        if isinstance(trace, dict) and not trace.get("ticket_file_names_row"):
            errors.append(f"{frz}: owner ticket file does not name the row")
        if (
            row.get("red_first_status") == "EVIDENCE_MISSING"
            and _GA2R_RECORD_NAME in str(row.get("red_first_evidence_ref") or "")
            and not str(row.get("red_first_reason") or "").strip()
        ):
            errors.append(f"{frz}: EVIDENCE_MISSING in the GA.2R record needs red_first_reason")
        if row.get("regression_scope") in (None, "", "MAPPING_MISSING"):
            errors.append(f"{frz}: missing regression_scope mapping")
        status = row.get("red_first_status")
        if status not in (
            "EVIDENCE_MISSING",
            "RED_AT_PREFIX",
            "GREEN_AT_PREFIX",
            "RED_FIRST_UNREPRODUCIBLE",
            "RED_FIRST_OUT",
            "RED_FIRST_LATER_WAVE",
        ):
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

    deps = data.get("dependencies")
    errors.extend(_validate_scope_dependencies(deps))
    errors.extend(_validate_lane_order(data.get("lane_order") or []))
    errors.extend(_train_errors(data, build))
    errors.extend(build.find_af2_errors(data.get("find_af2")))
    if open_outside_inventory is not None:
        errors.extend(_coverage_errors(data, open_outside_inventory))
    if isinstance(data.get("post_freeze_policy_schedule"), dict):
        errors.extend(_validate_evidence_missing_obligations(inventory))
        errors.extend(_validate_post_freeze_policy_schedule(data))
    return errors


_OPERATOR_CITATION = "eba9bb47a3bb282c1e001c9a20ad8ed6b596ddbb"
_OPERATOR_PACKET = (
    "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_V5_RED_CLASSES_PACKET.md"
)
_AUTHORIZED_RED_FIRST_OUT = frozenset({"FRZ-PLG-02", "FRZ-PLG-07", "FRZ-RPL-04", "FRZ-REL-07", "FRZ-PRD-08"})
_AUTHORIZED_LATER_WAVE = {
    "FRZ-FRZ-10": {
        "owner_node": "T-B7-PFP-ACTIVATE",
        "owner_wave": "W5",
        "verification_node": "T-B7-PFP-VERIFY",
        "verification_wave": "W8",
    }
}


def _cited_operator_disposition(row: dict[str, object]) -> bool:
    return (
        row.get("operator_citation") == _OPERATOR_CITATION
        and row.get("operator_packet_ref") == _OPERATOR_PACKET
        and row.get("red_first_historical_status") == "RED_FIRST_UNREPRODUCIBLE"
    )


def validate_red_first_evidence(data: dict[str, object]) -> list[str]:
    errors: list[str] = []
    for row in data.get("code_fix_inventory") or []:
        if not isinstance(row, dict):
            continue
        frz = str(row.get("frz_id"))
        status = row.get("red_first_status")
        sha = row.get("red_first_prefix_sha")
        ref = row.get("red_first_evidence_ref")
        if status == "EVIDENCE_MISSING":
            errors.append(f"{frz}: red-first evidence not recorded (EVIDENCE_MISSING)")
            if sha:
                errors.append(f"{frz}: EVIDENCE_MISSING must not carry prefix SHA")
        if status == "GREEN_AT_PREFIX":
            errors.append(f"{frz}: GREEN_AT_PREFIX does not demonstrate red-first replay")
        if status == "RED_FIRST_UNREPRODUCIBLE":
            errors.append(f"{frz}: RED_FIRST_UNREPRODUCIBLE does not demonstrate red-first replay")
            if not sha or not isinstance(sha, str) or len(sha) != 40:
                errors.append(f"{frz}: RED_FIRST_UNREPRODUCIBLE without valid historical prefix SHA")
            if not ref:
                errors.append(f"{frz}: RED_FIRST_UNREPRODUCIBLE without evidence reference")
            reason = row.get("red_first_reason")
            if not reason or not str(reason).strip():
                errors.append(f"{frz}: RED_FIRST_UNREPRODUCIBLE without red_first_reason")
        if status == "RED_AT_PREFIX":
            if not sha or not isinstance(sha, str) or len(sha) != 40:
                errors.append(f"{frz}: RED_AT_PREFIX without valid historical prefix SHA")
            if not ref:
                errors.append(f"{frz}: RED_AT_PREFIX without evidence reference")
        if status in ("RED_FIRST_OUT", "RED_FIRST_LATER_WAVE"):
            if not sha or not isinstance(sha, str) or len(sha) != 40:
                errors.append(f"{frz}: {status} without valid historical prefix SHA")
            if not ref:
                errors.append(f"{frz}: {status} without evidence reference")
            reason = row.get("red_first_reason")
            if not reason or not str(reason).strip():
                errors.append(f"{frz}: {status} without historical red_first_reason")
            if not _cited_operator_disposition(row):
                errors.append(f"{frz}: {status} without the operator T-AF2-RED-CLASSES citation")
            if status == "RED_FIRST_OUT" and frz not in _AUTHORIZED_RED_FIRST_OUT:
                errors.append(f"{frz}: RED_FIRST_OUT is not an authorized T-AF2-RED-CLASSES row")
            if status == "RED_FIRST_LATER_WAVE":
                expected = _AUTHORIZED_LATER_WAVE.get(frz)
                if expected is None:
                    errors.append(f"{frz}: RED_FIRST_LATER_WAVE is not an authorized T-AF2-RED-CLASSES row")
                else:
                    for key, value in expected.items():
                        if row.get(key) != value:
                            errors.append(f"{frz}: RED_FIRST_LATER_WAVE {key} must stay {value}")
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
    named_negative_tests: tuple[int, int] = (0, 0),
) -> None:
    for e in plan_errors:
        print(f"[PLAN_STRUCTURE] {e}", file=sys.stderr)
    print(f"PLAN_STRUCTURE={plan_status}")
    print(f"CODE_FIX_OWNERSHIP={ownership_verified}/{ownership_total}")
    print(f"NAMED_NEGATIVE_TESTS={named_negative_tests[0]}/{named_negative_tests[1]}")
    for e in red_first:
        print(f"[RED_FIRST_EVIDENCE] {e}")
    if red_first:
        print("RED_FIRST_EVIDENCE=INCOMPLETE")
    else:
        print("RED_FIRST_EVIDENCE=COMPLETE")
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
    open_rows = {
        str(r["frz_id"]) for r in _load_build_rw_plan().parse_appendix_a_all_rows() if r["matrix_status"] == "OPEN"
    }
    plan_errors.extend(validate_plan_structure(data, canonical, open_outside_inventory=open_rows - canonical))
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
        named_negative_tests=count_named_negative_tests(data),
    )
    return 0 if plan_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
