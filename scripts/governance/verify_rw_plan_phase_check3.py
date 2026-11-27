#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Strict WP-12 phase check 3 over RW-PLAN + FRZ matrix (mechanical; W2-RECOVERY-02)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
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
_GA2R_SCHEMA_ID = "af2.red_first_ga2r.v1"
_GA2R_JSON = (
    _REPO / "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json"
)
_AUTHORIZED_EVIDENCE_PREFIX = "docs/project/maintainers/qualification/arch_freeze_program/continuation/"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_INVALID_PREFIX_FAILURE_MARKERS = (
    "ModuleNotFoundError",
    "ImportError",
    "ERROR collecting",
    "errors during collection",
    "No module named",
    "timed out",
    "TimeoutError",
    "installation error",
    "Could not install",
)

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
_OD_AF_CLOSE_B6 = "OD-AF-CLOSE-B6"
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
    "op_b_pre_g2_enforcement": _OD_AF_CLOSE_B6,
}
_SEEDED_RED_REQUIRED = frozenset({"FRZ-DEBT-02", "FRZ-DEBT-03"})
_OP_B_ALLOWED_STATUSES = frozenset({"EVIDENCE_MISSING", "RED_AT_PREFIX"})
_OP_B_FORBIDDEN_STATUSES = frozenset({"RED_FIRST_OUT", "RED_FIRST_LATER_WAVE", "PASS"})


def _validate_op_b_row_metadata(frz: str, row: dict[str, object], errors: list[str]) -> None:
    expected_owner = _EVIDENCE_MISSING_OP_B[frz]
    if row.get("owner_ticket") != expected_owner:
        errors.append(f"{frz}: OP-B owner must be {expected_owner}")
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


def _validate_evidence_missing_obligations(inventory: list[object]) -> list[str]:
    """OP-B (OD-AF-CLOSE-B2): twelve rows stay identifiable; EVIDENCE_MISSING or RED_AT_PREFIX."""
    errors: list[str] = []
    by_frz: dict[str, dict[str, object]] = {}
    for row in inventory:
        if isinstance(row, dict) and row.get("frz_id"):
            by_frz[str(row["frz_id"])] = row
    for frz in _EVIDENCE_MISSING_OP_B:
        row = by_frz.get(frz)
        if row is None:
            errors.append(f"{frz}: OP-B obligation row missing from inventory")
            continue
        status = row.get("red_first_status")
        if status in _OP_B_FORBIDDEN_STATUSES:
            errors.append(f"{frz}: OP-B row must not use status {status}")
        elif status not in _OP_B_ALLOWED_STATUSES:
            errors.append(f"{frz}: OP-B row status must be EVIDENCE_MISSING or RED_AT_PREFIX, got {status!r}")
        else:
            _validate_op_b_row_metadata(frz, row, errors)
    return errors


def validate_pre_g2_op_b_settlement(
    inventory: list[object],
    *,
    data: dict[str, object] | None = None,
    git_validate: bool = True,
) -> list[str]:
    """Fail-closed OP-B settlement for pre-G2 / G2 qualification (OD-AF-CLOSE-B6).

    Independent of ``post_freeze_policy_schedule.pre_g2_op_b_gate_enforce`` when invoked from ``--pre-g2``.
    """
    errors: list[str] = []
    by_frz: dict[str, dict[str, object]] = {}
    for row in inventory:
        if not isinstance(row, dict):
            continue
        frz = row.get("frz_id")
        if not frz:
            continue
        key = str(frz)
        if key not in _EVIDENCE_MISSING_OP_B:
            continue
        if key in by_frz:
            errors.append(f"{key}: duplicate OP-B obligation row")
        by_frz[key] = row

    for frz in sorted(_EVIDENCE_MISSING_OP_B):
        row = by_frz.get(frz)
        if row is None:
            errors.append(f"{frz}: OP-B obligation row missing from inventory")
            continue
        status = row.get("red_first_status")
        if status in _OP_B_FORBIDDEN_STATUSES:
            errors.append(f"{frz}: OP-B row must not use status {status}")
        elif status != "RED_AT_PREFIX":
            errors.append(
                f"pre-G2 OP-B gate blocked ({_OD_AF_CLOSE_B6}): "
                f"{frz} requires RED_AT_PREFIX with complete evidence, got {status!r}"
            )
        _validate_op_b_row_metadata(frz, row, errors)
        if status == "RED_AT_PREFIX" and data is not None:
            errors.extend(_validate_op_b_red_at_prefix_completeness(frz, row, git_validate=git_validate))
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
    if schedule.get("pre_g2_op_b_gate_enforce") is True:
        errors.extend(
            validate_pre_g2_op_b_settlement(
                data.get("code_fix_inventory") or [],
                data=data,
            )
        )
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


def _valid_sha40(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA40.match(value))


def _valid_sha64(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _normalize_repo_relative(path_part: str) -> str | None:
    text = path_part.strip().replace("\\", "/")
    if not text or text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        return None
    if ".." in text.split("/"):
        return None
    parts = [p for p in text.split("/") if p not in ("", ".")]
    return "/".join(parts)


def _is_authorized_evidence_path(rel: str) -> bool:
    if not rel.endswith(".json"):
        return False
    return rel.startswith(_AUTHORIZED_EVIDENCE_PREFIX)


def _git_tracked(rel_path: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_path],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _git_object_exists(sha: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _git_is_ancestor(ancestor: str, descendant: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _git_blob_exists(blob_sha: str) -> bool:
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{blob_sha}^{{blob}}"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.returncode == 0
    except OSError:
        return False


def _git_blob_for_path_at_commit(commit: str, rel_path: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", f"{commit}:{rel_path}"],
            cwd=_REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        text = proc.stdout.strip()
        return text if _valid_sha40(text) else None
    except OSError:
        return None


def _pytest_node_test_path(node_id: str) -> str | None:
    text = node_id.strip()
    if not text:
        return None
    if "::" in text:
        return text.split("::", 1)[0]
    return text if text.endswith(".py") else None


def _overlay_test_blob_for_prefix(
    evidence_payload: dict[str, object], prefix_sha: str, rel_path: str
) -> tuple[str | None, str | None]:
    """Return (overlay_fix_commit, overlay_test_blob_sha) when prefix used test-only overlay."""
    runs = evidence_payload.get("prefix_runs")
    if not isinstance(runs, list):
        return None, None
    for entry in runs:
        if not isinstance(entry, dict) or str(entry.get("prefix_sha")) != prefix_sha:
            continue
        overlay = entry.get("overlay")
        if not isinstance(overlay, dict):
            continue
        meta = overlay.get(rel_path)
        if not isinstance(meta, dict):
            continue
        fix = meta.get("fix_commit")
        blob = meta.get("test_blob_sha")
        if _valid_sha40(fix) and _valid_sha40(blob):
            return str(fix), str(blob)
    return None, None


def _parse_evidence_ref(ref: object) -> tuple[str | None, str | None]:
    if not ref or not isinstance(ref, str):
        return None, None
    text = ref.strip()
    if "#" not in text:
        return None, None
    path_part, anchor = text.split("#", 1)
    if not path_part or not anchor or anchor.strip() != anchor:
        return None, None
    rel = _normalize_repo_relative(path_part)
    if rel is None:
        return None, None
    candidate = (_REPO / rel).resolve()
    try:
        candidate.relative_to(_REPO.resolve())
    except ValueError:
        return None, None
    return rel, anchor


def _load_ga2r_index_from_file(path: Path) -> tuple[dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return {}, ["evidence file missing"]
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, [f"evidence file read failed: {exc}"]
    except UnicodeError as exc:
        return {}, [f"evidence file encoding invalid: {exc}"]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, [f"evidence file invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return {}, ["evidence file schema: root must be an object"]
    schema = payload.get("schema")
    if schema != _GA2R_SCHEMA_ID:
        return {}, [f"evidence file schema: expected {_GA2R_SCHEMA_ID!r}, got {schema!r}"]
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return {}, ["evidence file schema: rows must be a list"]
    index: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("evidence file schema: row must be an object")
            continue
        frz_id = row.get("frz_id")
        if not frz_id:
            errors.append("evidence file schema: row missing frz_id")
            continue
        key = str(frz_id)
        if key in index:
            errors.append(f"evidence file duplicate frz_id {key}")
        index[key] = row
    return index, errors


def _load_ga2r_index(path: Path | None = None) -> dict[str, dict[str, object]]:
    target = path or _GA2R_JSON
    index, _errors = _load_ga2r_index_from_file(target)
    return index


def _node_matches_binding(node: str, binding: str) -> bool:
    if node == binding:
        return True
    if "::" not in binding and node.startswith(binding.rstrip("/")):
        return True
    return False


def _bound_ga2r_node(record: dict[str, object], binding: str) -> dict[str, object] | None:
    nodes = record.get("nodes")
    if not isinstance(nodes, list):
        return None
    for item in nodes:
        if not isinstance(item, dict):
            continue
        node = str(item.get("node") or "")
        if _node_matches_binding(node, binding):
            return item
    return None


def _assertion_is_mechanism_failure(assertion_lines: object) -> bool:
    if not isinstance(assertion_lines, list) or not assertion_lines:
        return False
    joined = "\n".join(str(line) for line in assertion_lines)
    if any(marker in joined for marker in _INVALID_PREFIX_FAILURE_MARKERS):
        return False
    lowered = joined.lower()
    return "assertionerror" in lowered


def _validate_seeded_red_record(frz: str, node: dict[str, object], binding: str, errors: list[str]) -> None:
    if node.get("seeded_red") is not True:
        errors.append(f"{frz}: OP-B seeded-red requires seeded_red=true plus full execution record")
        return
    executed = str(node.get("node") or "").strip()
    seeded_node = str(node.get("seeded_negative_node") or "").strip()
    if not seeded_node:
        errors.append(f"{frz}: OP-B seeded-red missing seeded_negative_node")
    elif seeded_node != executed:
        errors.append(f"{frz}: OP-B seeded-red seeded_negative_node must match executed node")
    elif not _node_matches_binding(executed, binding):
        errors.append(f"{frz}: OP-B seeded-red executed node must match planned negative test binding")
    if not str(node.get("seeded_violation_description") or "").strip():
        errors.append(f"{frz}: OP-B seeded-red missing seeded_violation_description")
    desc = str(node.get("seeded_violation_description") or "").lower()
    if frz == "FRZ-DEBT-02" and not any(token in desc for token in ("core", "ownership", "dependency")):
        errors.append(f"{frz}: OP-B seeded-red violation must describe unauthorized core dependency ownership")
    if frz == "FRZ-DEBT-03" and not any(token in desc for token in ("seam", "compatibility")):
        errors.append(f"{frz}: OP-B seeded-red violation must describe a temporary compatibility seam")


def _validate_op_b_test_version_integrity(
    frz: str,
    node: dict[str, object],
    binding: str,
    evidence_payload: dict[str, object],
    errors: list[str],
    *,
    git_validate: bool,
) -> None:
    test_blob = node.get("test_blob_sha")
    pin_blob = node.get("pin_blob_sha")
    if not _valid_sha40(test_blob) or not _valid_sha40(pin_blob):
        return
    if str(test_blob) != str(pin_blob):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX requires identical test version in red and green runs")
        return
    rel_path = _pytest_node_test_path(binding)
    if rel_path is None:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX negative test binding is not a pytest node id")
        return
    executed = str(node.get("node") or "")
    if executed and not _node_matches_binding(executed, binding):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX executed node must match bound negative test")
    if not git_validate:
        return
    pin_sha = node.get("pin_sha")
    prefix_sha = node.get("prefix_sha")
    if not _valid_sha40(pin_sha):
        return
    if not _git_blob_exists(str(test_blob)):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX test_blob_sha {test_blob} not found in repository")
    pin_tree_blob = _git_blob_for_path_at_commit(str(pin_sha), rel_path)
    if pin_tree_blob is None:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX test file {rel_path!r} missing at pin_sha")
    elif pin_tree_blob != str(pin_blob):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX pin_blob_sha does not match test file at pin_sha")
    if _valid_sha40(prefix_sha):
        prefix_tree_blob = _git_blob_for_path_at_commit(str(prefix_sha), rel_path)
        if prefix_tree_blob is None:
            overlay_fix, overlay_blob = _overlay_test_blob_for_prefix(evidence_payload, str(prefix_sha), rel_path)
            if overlay_fix is None or overlay_blob is None:
                errors.append(f"{frz}: OP-B RED_AT_PREFIX test file {rel_path!r} missing at prefix_sha without overlay")
            else:
                if overlay_blob != str(test_blob):
                    errors.append(f"{frz}: OP-B RED_AT_PREFIX overlay test_blob_sha mismatch")
                overlay_git_blob = _git_blob_for_path_at_commit(overlay_fix, rel_path)
                if overlay_git_blob is None:
                    errors.append(f"{frz}: OP-B RED_AT_PREFIX overlay fix_commit missing overlaid test file")
                elif overlay_git_blob != str(test_blob):
                    errors.append(f"{frz}: OP-B RED_AT_PREFIX overlay blob does not match test_blob_sha")
        elif prefix_tree_blob != str(test_blob):
            errors.append(f"{frz}: OP-B RED_AT_PREFIX test_blob_sha does not match test file at prefix_sha")


def _validate_op_b_execution_graph(frz: str, node: dict[str, object], errors: list[str], *, git_validate: bool) -> None:
    prefix_sha = node.get("prefix_sha")
    fix_commit = node.get("fix_commit")
    pin_sha = node.get("pin_sha")
    if not _valid_sha40(pin_sha):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing green-after-fix execution SHA (pin_sha)")
        return
    if _valid_sha40(prefix_sha) and _valid_sha40(fix_commit) and str(prefix_sha) == str(fix_commit):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX prefix_sha must not equal fix_commit")
    if not git_validate:
        return
    for label, sha in (("prefix_sha", prefix_sha), ("fix_commit", fix_commit), ("pin_sha", pin_sha)):
        if isinstance(sha, str) and _valid_sha40(sha) and not _git_object_exists(sha):
            errors.append(f"{frz}: OP-B RED_AT_PREFIX {label} {sha} not found in repository")
    if _valid_sha40(prefix_sha) and _valid_sha40(fix_commit) and _git_object_exists(str(prefix_sha)):
        if not _git_is_ancestor(str(prefix_sha), str(fix_commit)):
            errors.append(f"{frz}: OP-B RED_AT_PREFIX fix_commit must descend from prefix_sha")
    if _valid_sha40(fix_commit) and _valid_sha40(pin_sha) and _git_object_exists(str(fix_commit)):
        if not _git_is_ancestor(str(fix_commit), str(pin_sha)):
            errors.append(f"{frz}: OP-B RED_AT_PREFIX pin_sha must contain fix_commit in ancestry")


def _validate_op_b_red_at_prefix_completeness(
    frz: str,
    row: dict[str, object],
    *,
    git_validate: bool = True,
) -> list[str]:
    """Progressive OP-B contract (AF-W2-FINAL-01 / AF-W2-CLOSURE-04): bind evidence_ref to the cited JSON file."""
    errors: list[str] = []
    sha = row.get("red_first_prefix_sha")
    ref = row.get("red_first_evidence_ref")
    if not _valid_sha40(sha):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX requires full 40-char lowercase prefix SHA")
        return errors
    rel, anchor = _parse_evidence_ref(ref)
    if rel is None or anchor != frz:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX evidence ref must cite a repo-relative JSON file#FRZ anchor")
        return errors
    if not _is_authorized_evidence_path(rel):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX evidence path not in authorized continuation evidence")
        return errors
    if not _git_tracked(rel):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX evidence file must be git-tracked")
        return errors
    evidence_path = _REPO / rel
    try:
        evidence_raw = evidence_path.read_text(encoding="utf-8")
        evidence_payload = json.loads(evidence_raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX evidence incomplete ({exc})")
        return errors
    if not isinstance(evidence_payload, dict):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX evidence incomplete (root must be object)")
        return errors
    file_index, load_errors = _load_ga2r_index_from_file(evidence_path)
    for detail in load_errors:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX evidence incomplete ({detail})")
    if errors:
        return errors
    record = file_index.get(frz)
    if record is None:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX without GA2R inventory row in cited evidence file")
        return errors
    if str(record.get("frz_id")) != frz:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX GA2R frz_id mismatch in cited evidence file")
    binding = str(row.get("evidence_negative_test") or "").strip()
    if not binding:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing negative test binding")
        return errors
    node = _bound_ga2r_node(record, binding)
    if node is None:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing GA2R node for binding {binding!r}")
        return errors
    if not node.get("counted"):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX node is not counted for red-first replay")
    prefix_sha = node.get("prefix_sha")
    if prefix_sha != sha:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX inventory SHA must match bound node prefix_sha")
    fix_commit = node.get("fix_commit")
    if not _valid_sha40(fix_commit):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing fix_commit SHA")
    if node.get("prefix_result") != "FAILED":
        errors.append(f"{frz}: OP-B RED_AT_PREFIX requires FAILED at prefix (not collection/import noise)")
    if node.get("pin_result") != "PASSED":
        errors.append(f"{frz}: OP-B RED_AT_PREFIX requires PASSED after fix")
    if node.get("prefix_result") == "PASSED" and node.get("pin_result") == "PASSED":
        errors.append(f"{frz}: OP-B RED_AT_PREFIX cannot be both green at prefix and after fix")
    prefix_exit = node.get("prefix_run_exit")
    pin_exit = node.get("pin_run_exit")
    if prefix_exit is None or pin_exit is None:
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing isolated run exit codes")
    else:
        if prefix_exit != 1:
            errors.append(f"{frz}: OP-B RED_AT_PREFIX requires prefix_run_exit=1")
        if pin_exit != 0:
            errors.append(f"{frz}: OP-B RED_AT_PREFIX requires pin_run_exit=0")
    if not _assertion_is_mechanism_failure(node.get("assertion_lines")):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX lacks mechanism-aligned assertion failure at prefix")
    if not _valid_sha40(node.get("test_blob_sha")):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing executed test version digest (test_blob_sha)")
    if not _valid_sha40(node.get("pin_blob_sha")):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing post-fix test version digest (pin_blob_sha)")
    if not _valid_sha64(node.get("prefix_log_sha256")):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing prefix measurement log digest (prefix_log_sha256)")
    if not _valid_sha64(node.get("pin_log_sha256")):
        errors.append(f"{frz}: OP-B RED_AT_PREFIX missing green measurement log digest (pin_log_sha256)")
    _validate_op_b_test_version_integrity(frz, node, binding, evidence_payload, errors, git_validate=git_validate)
    _validate_op_b_execution_graph(frz, node, errors, git_validate=git_validate)
    if frz in _SEEDED_RED_REQUIRED and row.get("evidence_seeded_red_required"):
        _validate_seeded_red_record(frz, node, binding, errors)
    return errors


def _cited_operator_disposition(row: dict[str, object]) -> bool:
    return (
        row.get("operator_citation") == _OPERATOR_CITATION
        and row.get("operator_packet_ref") == _OPERATOR_PACKET
        and row.get("red_first_historical_status") == "RED_FIRST_UNREPRODUCIBLE"
    )


def validate_red_first_evidence(
    data: dict[str, object],
    *,
    ga2r_index: dict[str, dict[str, object]] | None = None,
    git_validate: bool = True,
) -> list[str]:
    errors: list[str] = []
    _ = ga2r_index  # injection reserved for isolated unit tests; production loads cited files only
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
            if not _valid_sha40(sha):
                errors.append(f"{frz}: RED_AT_PREFIX without valid historical prefix SHA")
            if not ref:
                errors.append(f"{frz}: RED_AT_PREFIX without evidence reference")
            elif sha and not _SHA40.match(str(sha)):
                errors.append(f"{frz}: RED_AT_PREFIX prefix SHA has invalid characters")
            elif frz in _EVIDENCE_MISSING_OP_B and _valid_sha40(sha) and ref:
                errors.extend(_validate_op_b_red_at_prefix_completeness(frz, row, git_validate=git_validate))
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict WP-12 phase check 3 over RW-PLAN + FRZ matrix.")
    parser.add_argument(
        "--pre-g2",
        action="store_true",
        help="Engage mandatory OP-B settlement checks for pre-G2 / G2 qualification (OD-AF-CLOSE-B6).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan_errors: list[str] = []
    if not _RW_JSON.is_file():
        plan_errors.append(f"missing {_RW_JSON.relative_to(_REPO)}")
        _emit(plan_errors, [], ["audit blocked: missing rw plan"], "FAIL", ownership_verified=0, ownership_total=0)
        if args.pre_g2:
            print("PRE_G2_OP_B_SETTLEMENT=FAIL")
        return 1

    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    try:
        canonical = _canonical_code_fix_ids()
    except SystemExit as exc:
        plan_errors.append(f"matrix parse failed: {exc}")
        _emit(plan_errors, [], ["audit blocked: matrix parse failed"], "FAIL", ownership_verified=0, ownership_total=0)
        if args.pre_g2:
            print("PRE_G2_OP_B_SETTLEMENT=FAIL")
        return 1

    ownership_verified, ownership_total = count_verified_ownership(data)
    open_rows = {
        str(r["frz_id"]) for r in _load_build_rw_plan().parse_appendix_a_all_rows() if r["matrix_status"] == "OPEN"
    }
    plan_errors.extend(validate_plan_structure(data, canonical, open_outside_inventory=open_rows - canonical))
    red_first_errors = validate_red_first_evidence(data)
    audit_notes = validate_rw_plan_audit(data)
    plan_status = "PASS" if not plan_errors else "FAIL"

    pre_g2_errors: list[str] = []
    pre_g2_settlement = ""
    if args.pre_g2:
        pre_g2_errors = validate_pre_g2_op_b_settlement(
            data.get("code_fix_inventory") or [],
            data=data,
        )
        for e in pre_g2_errors:
            print(f"[PRE_G2_OP_B] {e}", file=sys.stderr)
        pre_g2_settlement = "PASS" if plan_status == "PASS" and not pre_g2_errors else "FAIL"
        print(f"PRE_G2_OP_B_SETTLEMENT={pre_g2_settlement}")

    _emit(
        plan_errors,
        red_first_errors,
        audit_notes,
        plan_status,
        ownership_verified=ownership_verified,
        ownership_total=ownership_total,
        named_negative_tests=count_named_negative_tests(data),
    )
    if args.pre_g2:
        return 0 if pre_g2_settlement == "PASS" else 1
    return 0 if plan_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
