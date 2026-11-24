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
            {"finding_id": "A4-006", "owner_ticket": "T-IGX-044-A", "disposition": "x"},
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
                "T-AF2-RW-PLAN",
                "T-AF2-RW-PLAN-AUDIT",
                "PT-V04",
                "PT-B01",
                "PT-B02",
                "REV-MASTER",
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
    assert any("missing owner_ticket" in e for e in errors)


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
    assert any("not established" in n for n in audit)


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
    assert "RW_PLAN_AUDIT=INCOMPLETE" in proc.stdout
