# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
_VERIFY = _REPO / "scripts/governance/verify_rw_plan_phase_check3.py"


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_rw_plan_phase_check3", _VERIFY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _base_plan_row(frz: str = "FRZ-BND-04") -> dict:
    return {
        "frz_id": frz,
        "summary": "s",
        "owner_ticket": "T-IGX-043-G",
        "negative_test_id": "NT-BND-04",
        "negative_tests": _negative_tests(),
        "red_first_prefix_sha": None,
        "red_first_status": "EVIDENCE_MISSING",
        "regression_scope": "ENT-A-BND-01..03",
        "remediation_track": "CODE_FIX:",
    }


def _negative_tests() -> list[dict]:
    return [
        {
            "nt_id": "NT-BND-04",
            "node": "tests/unit/test_x.py::test_a",
            "mechanism": "m",
            "owner_ticket": "T-IGX-043-G",
            "file_state_at_pin": "exists",
        }
    ]


def _minimal_plan(extra_rows: list[dict] | None = None) -> dict:
    rows = [_base_plan_row()]
    if extra_rows:
        rows.extend(extra_rows)
    return {
        "schema": "rw_plan.v1",
        "code_fix_inventory": rows,
        "audit_flag_conflicts": [
            {"finding_id": "A4-005", "owner_ticket": "T-IGX-051-H", "disposition": "x"},
            {"finding_id": "A4-018", "owner_ticket": "T-IGX-051-D", "disposition": "x"},
            {"finding_id": "A6-005", "owner_ticket": "T-IGX-043-I", "disposition": "x"},
            {
                "finding_id": "A4-006",
                "owner_ticket": "T-IGX-078-D",
                "owner_node": "PT-T01",
                "owner_wave": "W3",
                "owner_source": "WP12 DAG / PT-T01 / A4-006; GA.2R T-IGX-078-D",
                "disposition": "x",
            },
            {"finding_id": "A3-018", "owner_ticket": "T-IGX-046-G", "disposition": "x"},
            {"finding_id": "A4-016", "owner_ticket": "T-IGX-043-H", "disposition": "x"},
        ],
        "lane_order": [
            ("lane_1", ["T-IGX-043-G"], "PT-WA"),
            ("lane_2", ["T-IGX-043-H"], "PT-WB"),
        ],
        "tickets": [
            {
                "ticket": ticket,
                "lifecycle": "plan_approved",
                "landed": False,
                "write_set": [{"path": "tests/unit/test_x.py", "state": "exists"}],
            }
            for ticket in ("T-IGX-043-G", "T-IGX-043-H")
        ],
        "overlap_matrix": [{"a": "T-IGX-043-G", "b": "T-IGX-043-H", "paths": ["tests/unit/test_x.py"]}],
        "serialization_edges": [{"first": "T-IGX-043-G", "second": "T-IGX-043-H", "reason": "c"}],
        "coverage": [],
        "find_af2": [{"id": f"FIND-AF2-{n:02d}", "mapping": "node", "in_train": False} for n in range(1, 24)],
        "dependencies": {
            "W2": [],
            "SCOPE": [
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
            ],
            "W3": [],
        },
    }


def test_rw_plan_json_exists_and_schema():
    assert _RW_JSON.is_file(), "run scripts/governance/build_rw_plan.py first"
    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    assert data.get("schema") == "rw_plan.v1"
    assert len(data.get("code_fix_inventory") or []) >= 35


def test_missing_owner_detected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].pop("owner_ticket")
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e for e in errors)


def test_missing_regression_mapping():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["regression_scope"] = "MAPPING_MISSING"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("regression_scope" in e for e in errors)


def test_missing_audit_conflict():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["audit_flag_conflicts"] = []
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("audit_flag_conflicts missing" in e for e in errors)


def test_duplicate_audit_conflict():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["audit_flag_conflicts"].append(plan["audit_flag_conflicts"][0].copy())
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("duplicate audit conflict" in e for e in errors)


def test_reversed_pt_wa_pt_wb():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["lane_order"] = [
        ("lane_1", ["T-IGX-043-H"], "PT-WB"),
        ("lane_2", ["T-IGX-043-G"], "PT-WA"),
    ]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("PT-WB precedes PT-WA" in e for e in errors)


def test_mechanical_pass_does_not_imply_audit_pass():
    mod = _load_verify()
    plan = _minimal_plan()
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert not errors
    audit = mod.validate_rw_plan_audit(plan)
    assert any("not performed" in n for n in audit)


def test_red_first_evidence_missing_sha_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["red_first_prefix_sha"] = "a" * 40
    errors = mod.validate_red_first_evidence(plan)
    assert any("EVIDENCE_MISSING must not carry prefix SHA" in e for e in errors)


def test_red_at_prefix_requires_sha_and_ref():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["red_first_status"] = "RED_AT_PREFIX"
    errors = mod.validate_red_first_evidence(plan)
    assert any("prefix SHA" in e for e in errors)
    assert any("evidence reference" in e for e in errors)


def test_red_first_unreproducible_accepted_by_structure():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].update(
        {
            "red_first_status": "RED_FIRST_UNREPRODUCIBLE",
            "red_first_prefix_sha": "a" * 40,
            "red_first_evidence_ref": "docs/x.json#FRZ-BND-04",
            "red_first_reason": "strict exit 2",
        }
    )
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert not errors


def test_authorized_out_is_not_red_at_prefix():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].update(
        {
            "frz_id": "FRZ-PLG-02",
            "red_first_status": "RED_FIRST_OUT",
            "red_first_historical_status": "RED_FIRST_UNREPRODUCIBLE",
            "red_first_prefix_sha": "b" * 40,
            "red_first_evidence_ref": "docs/packet#FRZ-PLG-02",
            "red_first_reason": "pytest exit 0 at prefix",
            "operator_citation": mod._OPERATOR_CITATION,
            "operator_packet_ref": mod._OPERATOR_PACKET,
        }
    )
    errors = mod.validate_red_first_evidence(plan)
    assert errors == []
    assert plan["code_fix_inventory"][0]["red_first_status"] != "RED_AT_PREFIX"


def test_out_without_operator_citation_stays_incomplete():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].update(
        {
            "frz_id": "FRZ-PLG-02",
            "red_first_status": "RED_FIRST_OUT",
            "red_first_prefix_sha": "b" * 40,
            "red_first_evidence_ref": "docs/x.md",
            "red_first_reason": "historical",
        }
    )
    errors = mod.validate_red_first_evidence(plan)
    assert any("without the operator T-AF2-RED-CLASSES citation" in e for e in errors)


def test_out_on_unauthorized_row_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].update(
        {
            "frz_id": "FRZ-BND-04",
            "red_first_status": "RED_FIRST_OUT",
            "red_first_historical_status": "RED_FIRST_UNREPRODUCIBLE",
            "red_first_prefix_sha": "b" * 40,
            "red_first_evidence_ref": "docs/x.md",
            "red_first_reason": "historical",
            "operator_citation": mod._OPERATOR_CITATION,
            "operator_packet_ref": mod._OPERATOR_PACKET,
        }
    )
    errors = mod.validate_red_first_evidence(plan)
    assert any("not an authorized T-AF2-RED-CLASSES row" in e for e in errors)


def test_later_wave_requires_w5_w8_owners():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].update(
        {
            "frz_id": "FRZ-FRZ-10",
            "red_first_status": "RED_FIRST_LATER_WAVE",
            "red_first_historical_status": "RED_FIRST_UNREPRODUCIBLE",
            "red_first_prefix_sha": "c" * 40,
            "red_first_evidence_ref": "docs/x.md#FRZ-FRZ-10",
            "red_first_reason": "AttributeError",
            "operator_citation": mod._OPERATOR_CITATION,
            "operator_packet_ref": mod._OPERATOR_PACKET,
            "owner_node": "T-B7-PFP-ACTIVATE",
            "owner_wave": "W5",
            "verification_node": "T-B7-PFP-VERIFY",
            "verification_wave": "W8",
        }
    )
    assert mod.validate_red_first_evidence(plan) == []
    plan["code_fix_inventory"][0]["owner_wave"] = "W3"
    errors = mod.validate_red_first_evidence(plan)
    assert any("owner_wave must stay W5" in e for e in errors)


def test_red_first_unreproducible_incomplete_evidence():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0].update(
        {
            "red_first_status": "RED_FIRST_UNREPRODUCIBLE",
            "red_first_prefix_sha": "a" * 40,
            "red_first_evidence_ref": "docs/x.json#FRZ-BND-04",
            "red_first_reason": "strict exit 2",
        }
    )
    errors = mod.validate_red_first_evidence(plan)
    assert any("does not demonstrate red-first replay" in e for e in errors)


def test_evidence_missing_rows_report_incomplete():
    mod = _load_verify()
    plan = _minimal_plan()
    errors = mod.validate_red_first_evidence(plan)
    assert any("EVIDENCE_MISSING" in e for e in errors)


def test_frz_frz_10_node_ownership_accepted():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0] = {
        "frz_id": "FRZ-FRZ-10",
        "summary": "s",
        "owner_ticket": None,
        "owner_node": "T-B7-PFP-ACTIVATE",
        "owner_wave": "W5",
        "verification_node": "T-B7-PFP-VERIFY",
        "verification_wave": "W8",
        "owner_source": "MEP-AF2 v2.1 / W5 and W8",
        "owner_ticket_family": "T-IGX-052-*",
        "owner_provenance": "explicit_mep_wp12",
        "negative_test_id": "NT-FRZ-10",
        "negative_tests": _negative_tests(),
        "red_first_status": "EVIDENCE_MISSING",
        "regression_scope": "ENT-A",
        "remediation_track": "CODE_FIX:",
    }
    errors = mod.validate_plan_structure(plan, {"FRZ-FRZ-10"})
    assert not errors


def test_contradictory_ownership_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["owner_node"] = "T-AF2-RW-TYP"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e or "contradictory" in e for e in errors)


def test_ownership_validation_independent_of_red_first():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["red_first_prefix_sha"] = "a" * 40
    struct = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    red = mod.validate_red_first_evidence(plan)
    assert struct == []
    assert any("EVIDENCE_MISSING must not carry prefix SHA" in e for e in red)


def test_code_fix_ownership_count_on_built_plan():
    mod = _load_verify()
    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    verified, total = mod.count_verified_ownership(data)
    assert total == 39
    assert verified == 39


def test_phase_check3_cli_emits_separate_dimensions():
    proc = subprocess.run(
        [sys.executable, str(_VERIFY)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "PLAN_STRUCTURE=PASS" in proc.stdout, proc.stderr
    assert proc.returncode == 0
    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    expected = "INCOMPLETE" if _load_verify().validate_red_first_evidence(data) else "COMPLETE"
    assert f"RED_FIRST_EVIDENCE={expected}" in proc.stdout
    assert "NAMED_NEGATIVE_TESTS=39/39" in proc.stdout
    assert "RW_PLAN_AUDIT=NOT_PERFORMED" in proc.stdout


@pytest.mark.parametrize("identifier", ["NEG-0", "NEG-P", "gate extension", None])
def test_class_marker_is_not_a_negative_test_identifier(identifier):
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["negative_test_id"] = identifier
    plan["code_fix_inventory"][0]["remediation_track"] = "CODE_FIX: gate extension"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("negative_test_id" in e for e in errors)


@pytest.mark.parametrize("missing", ["nt_id", "node", "mechanism", "owner_ticket", "file_state_at_pin"])
def test_negative_test_entry_must_name_node_mechanism_owner(missing: str):
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["negative_tests"][0].pop(missing)
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("negative test entry" in e and missing in e for e in errors)


def test_row_without_negative_tests_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["negative_tests"] = []
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("negative_tests" in e for e in errors)


def test_named_negative_test_count():
    mod = _load_verify()
    plan = _minimal_plan()
    assert mod.count_named_negative_tests(plan) == (1, 1)
    plan["code_fix_inventory"][0]["negative_test_id"] = "NEG-0"
    assert mod.count_named_negative_tests(plan) == (0, 1)


def test_evidence_missing_requires_reason_in_ga2r_record():
    mod = _load_verify()
    plan = _minimal_plan()
    row = plan["code_fix_inventory"][0]
    row["red_first_evidence_ref"] = "docs/x/RED_FIRST_GA2R_RESULT.json#FRZ-BND-04"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("red_first_reason" in e for e in errors)
    row["red_first_reason"] = "node absent at pin"
    assert mod.validate_plan_structure(plan, {"FRZ-BND-04"}) == []


def test_overlap_pair_without_ordering_path_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["serialization_edges"] = []
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("without ordering" in e for e in errors)


def test_stale_overlap_matrix_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["overlap_matrix"] = []
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("overlap_matrix" in e for e in errors)


def test_lane_with_overlapping_members_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["lane_order"] = [("lane_1", ["T-IGX-043-G", "T-IGX-043-H"], "PT-WA"), ("lane_2", [], "PT-WB")]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("lane" in e and "T-IGX-043-H" in e for e in errors)


def test_missing_find_af2_mapping_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["find_af2"] = plan["find_af2"][:-1]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("FIND-AF2" in e for e in errors)


def test_open_row_without_coverage_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"}, open_outside_inventory={"FRZ-DOC-06"})
    assert any("coverage" in e and "FRZ-DOC-06" in e for e in errors)


def test_live_plan_binds_a_regression_command_to_every_row():
    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    missing = [r["frz_id"] for r in data["code_fix_inventory"] if "tests/" not in str(r.get("regression_command"))]
    assert not missing


def test_live_plan_covers_every_open_matrix_row():
    data = json.loads(_RW_JSON.read_text(encoding="utf-8"))
    build = _import_build_mod()
    rows = build.parse_appendix_a_all_rows()
    inventory = {r["frz_id"] for r in data["code_fix_inventory"]}
    open_outside = {r["frz_id"] for r in rows if r["matrix_status"] == "OPEN"} - inventory
    assert open_outside
    assert open_outside == {c["frz_id"] for c in data["coverage"]}


def _import_build_mod():
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_rw_plan", _REPO / "scripts/governance/build_rw_plan.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _explicit_row(frz: str, **overrides) -> dict:
    mod = _import_build_mod()
    contract = mod.EXPLICIT_CODE_FIX_OWNERSHIP[frz]
    base = {
        "frz_id": frz,
        "summary": "s",
        "negative_test_id": "NT-X",
        "red_first_status": "EVIDENCE_MISSING",
        "regression_scope": "ENT-A",
        "remediation_track": "CODE_FIX:",
        "negative_tests": _negative_tests(),
    }
    base.update(contract)
    base.update(overrides)
    return base


def test_frz_ten_044_a_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TEN-02", owner_ticket="T-IGX-044-A")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TEN-02"})
    assert any("FRZ-TEN-02" in e and "ownership" in e for e in errors)


def test_frz_ten_pt_t01_accepted():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TEN-02")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TEN-02"})
    assert not [e for e in errors if "FRZ-TEN-02" in e]


def test_a4_006_044_a_audit_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    for c in plan["audit_flag_conflicts"]:
        if c["finding_id"] == "A4-006":
            c["owner_ticket"] = "T-IGX-044-A"
            c.pop("owner_node", None)
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("A4-006" in e for e in errors)


def test_a4_006_pt_t01_audit_accepted():
    mod = _load_verify()
    plan = _minimal_plan()
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert not [e for e in errors if "A4-006" in e]


def test_incorrect_typ_ticket_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TYP-01", owner_ticket="T-IGX-043-G")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TYP-01"})
    assert any("FRZ-TYP-01" in e for e in errors)


def test_incorrect_typ_node_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TYP-01", owner_node="PT-T01")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TYP-01"})
    assert any("FRZ-TYP-01" in e for e in errors)


def test_arbitrary_ticket_cannot_replace_matrix_owner():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["owner_ticket"] = "T-IGX-051-A"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e for e in errors)


def test_arbitrary_node_outside_contract_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["owner_node"] = "PT-T01"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e for e in errors)


def test_missing_explicit_provenance_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TEN-02")])
    plan["code_fix_inventory"][1].pop("owner_provenance")
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TEN-02"})
    assert any("provenance" in e for e in errors)


def test_missing_scope_predecessor_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["dependencies"]["SCOPE"] = plan["dependencies"]["SCOPE"][:-1]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("SCOPE missing predecessors" in e for e in errors)


def test_non_list_scope_dependency_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["dependencies"]["SCOPE"] = "not-a-list"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("must be a list" in e for e in errors)


def test_duplicate_scope_predecessor_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["dependencies"]["SCOPE"] = plan["dependencies"]["SCOPE"] + ["M1"]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("duplicate predecessors" in e for e in errors)


def test_missing_pt_wa_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["lane_order"] = [e for e in plan["lane_order"] if e[2] != "PT-WA"]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("missing PT-WA" in e for e in errors)


def test_missing_pt_wb_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["lane_order"] = [e for e in plan["lane_order"] if e[2] != "PT-WB"]
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("missing PT-WB" in e for e in errors)


def test_duplicate_pt_wa_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["lane_order"].append(plan["lane_order"][0])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("duplicate PT-WA" in e for e in errors)


def test_canonical_scope_dependencies_accepted():
    mod = _load_verify()
    plan = _minimal_plan()
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert not [e for e in errors if "SCOPE" in e]


def test_unknown_frz_arbitrary_ticket_rejected():
    mod = _load_verify()
    plan = _minimal_plan(
        [
            {
                "frz_id": "FRZ-UNKNOWN-99",
                "summary": "s",
                "owner_ticket": "T-IGX-043-G",
                "negative_test_id": "NT-X",
                "red_first_status": "EVIDENCE_MISSING",
                "regression_scope": "ENT-A",
                "remediation_track": "CODE_FIX:",
            }
        ]
    )
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-UNKNOWN-99"})
    assert any("FRZ-UNKNOWN-99" in e and "ownership" in e for e in errors)


def test_canonical_frz_missing_matrix_owner_rejected():
    mod = _load_verify()
    build = _import_build_mod()
    matrix = {r.frz_id: r.owner_ticket for r in build.parse_appendix_a_code_fix_inventory()}
    frz = "FRZ-CANON-NULL-OWNER"
    matrix[frz] = None
    plan = _minimal_plan(
        [
            {
                "frz_id": frz,
                "summary": "s",
                "owner_ticket": "T-IGX-043-G",
                "negative_test_id": "NT-X",
                "red_first_status": "EVIDENCE_MISSING",
                "regression_scope": "ENT-A",
                "remediation_track": "CODE_FIX:",
            }
        ]
    )
    errors = mod.validate_plan_structure(
        plan,
        {"FRZ-BND-04", frz},
        matrix_owners=matrix,
        explicit=dict(build.EXPLICIT_CODE_FIX_OWNERSHIP),
    )
    assert any(frz in e and "ownership" in e for e in errors)


def test_canonical_frz_null_owner_without_contract_rejected():
    mod = _load_verify()
    build = _import_build_mod()
    matrix = {r.frz_id: r.owner_ticket for r in build.parse_appendix_a_code_fix_inventory()}
    frz = "FRZ-CANON-NULL-OWNER"
    matrix[frz] = None
    plan = _minimal_plan(
        [
            {
                "frz_id": frz,
                "summary": "s",
                "owner_ticket": None,
                "negative_test_id": "NT-X",
                "red_first_status": "EVIDENCE_MISSING",
                "regression_scope": "ENT-A",
                "remediation_track": "CODE_FIX:",
            }
        ]
    )
    errors = mod.validate_plan_structure(
        plan,
        {"FRZ-BND-04", frz},
        matrix_owners=matrix,
        explicit=dict(build.EXPLICIT_CODE_FIX_OWNERSHIP),
    )
    assert any(frz in e and "ownership" in e for e in errors)


def test_matrix_row_unauthorized_provenance_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["owner_provenance"] = "explicit_mep_wp12"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e for e in errors)


def test_matrix_row_unauthorized_owner_wave_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["owner_wave"] = "W3"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e for e in errors)


def test_matrix_row_unauthorized_verification_node_rejected():
    mod = _load_verify()
    plan = _minimal_plan()
    plan["code_fix_inventory"][0]["verification_node"] = "PT-T01"
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("ownership not verified" in e for e in errors)


def test_pt_t01_unexpected_verification_node_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TEN-02", verification_node="T-B7-PFP-VERIFY")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TEN-02"})
    assert any("FRZ-TEN-02" in e for e in errors)


def test_pt_t01_unexpected_verification_wave_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-TEN-02", verification_wave="W8")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TEN-02"})
    assert any("FRZ-TEN-02" in e for e in errors)


def test_explicit_contract_missing_null_owner_ticket_key_rejected():
    mod = _load_verify()
    row = _explicit_row("FRZ-TEN-02")
    row.pop("owner_ticket")
    plan = _minimal_plan([row])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-TEN-02"})
    assert any("FRZ-TEN-02" in e for e in errors)


def test_frz_frz_10_incorrect_owner_ticket_family_rejected():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-FRZ-10", owner_ticket_family="T-IGX-043-*")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-FRZ-10"})
    assert any("FRZ-FRZ-10" in e for e in errors)


def test_frz_frz_10_missing_owner_ticket_family_rejected():
    mod = _load_verify()
    row = _explicit_row("FRZ-FRZ-10")
    row.pop("owner_ticket_family")
    plan = _minimal_plan([row])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-FRZ-10"})
    assert any("FRZ-FRZ-10" in e for e in errors)


def test_frz_frz_10_complete_contract_accepted():
    mod = _load_verify()
    plan = _minimal_plan([_explicit_row("FRZ-FRZ-10")])
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04", "FRZ-FRZ-10"})
    assert not [e for e in errors if "FRZ-FRZ-10" in e]


@pytest.mark.parametrize(
    "scope_value",
    [
        [{}],
        [[]],
        [None],
        [1],
        [True],
        [""],
        [" "],
        [" M1"],
        ["M1 "],
    ],
)
def test_malformed_scope_entries_rejected(scope_value):
    mod = _load_verify()
    plan = _minimal_plan()
    plan["dependencies"]["SCOPE"] = scope_value
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("SCOPE" in e for e in errors)


@pytest.mark.parametrize("deps", [[], "not-a-dict"])
def test_non_dict_dependencies_rejected(deps):
    mod = _load_verify()
    plan = _minimal_plan()
    plan["dependencies"] = deps
    errors = mod.validate_plan_structure(plan, {"FRZ-BND-04"})
    assert any("dependencies" in e.lower() for e in errors)
