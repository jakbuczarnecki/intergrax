# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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
        "red_first_prefix_sha": None,
        "red_first_status": "EVIDENCE_MISSING",
        "regression_scope": "ENT-A-BND-01..03",
        "remediation_track": "CODE_FIX:",
    }


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
                "owner_ticket": None,
                "owner_node": "PT-T01",
                "owner_wave": "W3",
                "owner_source": "WP12 DAG / PT-T01 / A4-006",
                "disposition": "x",
            },
            {"finding_id": "A3-018", "owner_ticket": "T-IGX-046-G", "disposition": "x"},
            {"finding_id": "A4-016", "owner_ticket": "T-IGX-043-H", "disposition": "x"},
        ],
        "lane_order": [
            ("lane_1", "T-IGX-043-G", "PT-WA"),
            ("lane_2", "T-IGX-043-H", "PT-WB"),
        ],
        "serialization_pairs": [{"first": "a", "second": "b", "reason": "c"}],
        "overlap_hotspots": {"x": ["y"]},
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
        ("lane_1", "T-IGX-043-H", "PT-WB"),
        ("lane_2", "T-IGX-043-G", "PT-WA"),
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
        "owner_provenance": "explicit_mep_wp12",
        "negative_test_id": "NT-FRZ-10",
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
    assert "PLAN_STRUCTURE=" in proc.stdout
    assert "RED_FIRST_EVIDENCE=INCOMPLETE" in proc.stdout
    assert "RW_PLAN_AUDIT=NOT_PERFORMED" in proc.stdout


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
        "owner_provenance": "explicit_mep_wp12",
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
