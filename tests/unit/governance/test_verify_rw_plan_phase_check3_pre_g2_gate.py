# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_VERIFY = _REPO / "scripts/governance/verify_rw_plan_phase_check3.py"
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
_OP_B = (
    "FRZ-BND-05",
    "FRZ-BND-06",
    "FRZ-DEBT-02",
    "FRZ-DEBT-03",
    "FRZ-DEBT-04",
    "FRZ-EXE-02",
    "FRZ-PLG-05",
    "FRZ-SEC-06",
    "FRZ-TEN-02",
    "FRZ-TYP-01",
    "FRZ-TYP-03",
    "FRZ-TYP-05",
)


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_rw_plan_phase_check3", _VERIFY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _live_rw_plan() -> dict:
    return json.loads(_RW_JSON.read_text(encoding="utf-8"))


def _complete_all_op_b(plan: dict, mod) -> None:
    """Synthetic complete evidence only in tests."""
    from tests.unit.governance.test_verify_rw_plan_phase_check3_op_b_closure03 import (
        _complete_ga2r,
        _op_b_row,
        _patch_evidence_file,
    )

    ga2r_by_frz = {}
    for frz in _OP_B:
        row, binding = _op_b_row(plan, frz)
        row["red_first_status"] = "RED_AT_PREFIX"
        row["red_first_prefix_sha"] = "a" * 40
        row["red_first_evidence_ref"] = (
            f"docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json#{frz}"
        )
        ga2r_by_frz[frz] = _complete_ga2r(frz, binding)
    ref = "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json"
    with _patch_evidence_file(mod, {ref: ga2r_by_frz}):
        errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert not errors, errors


def test_default_mode_accepts_twelve_evidence_missing_with_structure_pass():
    proc = subprocess.run(
        [sys.executable, str(_VERIFY)],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "PLAN_STRUCTURE=PASS" in proc.stdout
    assert "RED_FIRST_EVIDENCE=INCOMPLETE" in proc.stdout


def test_pre_g2_mode_rejects_live_inventory():
    proc = subprocess.run(
        [sys.executable, str(_VERIFY), "--pre-g2"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "PRE_G2_OP_B_SETTLEMENT=FAIL" in proc.stdout


def test_pre_g2_strict_when_plan_flag_false():
    mod = _load_verify()
    plan = _live_rw_plan()
    plan["post_freeze_policy_schedule"]["pre_g2_op_b_gate_enforce"] = False
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert errors


def test_pre_g2_missing_row_fails():
    mod = _load_verify()
    plan = _live_rw_plan()
    plan["code_fix_inventory"] = [r for r in plan["code_fix_inventory"] if r["frz_id"] != "FRZ-BND-05"]
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert any("FRZ-BND-05" in e and "missing" in e for e in errors)


def test_pre_g2_duplicate_row_fails():
    mod = _load_verify()
    plan = _live_rw_plan()
    dup = copy.deepcopy(next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-BND-05"))
    plan["code_fix_inventory"].append(dup)
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert any("duplicate" in e for e in errors)


def test_pre_g2_forbidden_status_fails():
    mod = _load_verify()
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-BND-05")
    row["red_first_status"] = "RED_FIRST_OUT"
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert any("must not use status" in e for e in errors)


def test_pre_g2_red_at_prefix_without_evidence_fails():
    mod = _load_verify()
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-BND-05")
    row["red_first_status"] = "RED_AT_PREFIX"
    row["red_first_prefix_sha"] = "a" * 40
    row["red_first_evidence_ref"] = (
        "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json#FRZ-BND-05"
    )
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert any("OP-B RED_AT_PREFIX" in e or "without GA2R" in e for e in errors)


def test_pre_g2_unauthorized_owner_fails():
    mod = _load_verify()
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-BND-05")
    row["owner_ticket"] = "T-IGX-099-Z"
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert any("OP-B owner must be" in e for e in errors)


def test_pre_g2_seeded_red_debt02_flag_required():
    mod = _load_verify()
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-DEBT-02")
    row["evidence_seeded_red_required"] = False
    errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert any("seeded-red" in e for e in errors)


def test_pre_g2_synthetic_complete_twelve_passes_mechanical_contract():
    mod = _load_verify()
    plan = _live_rw_plan()
    from tests.unit.governance.test_verify_rw_plan_phase_check3_op_b_closure03 import (
        _complete_ga2r,
        _op_b_row,
        _patch_evidence_file,
    )

    ga2r_by_frz = {}
    for frz in _OP_B:
        row, binding = _op_b_row(plan, frz)
        row["red_first_status"] = "RED_AT_PREFIX"
        ga2r = _complete_ga2r(frz, binding)
        if frz == "FRZ-DEBT-02":
            row["evidence_seeded_red_required"] = True
            node = ga2r["nodes"][0]
            node.update(
                {
                    "seeded_red": True,
                    "seeded_negative_node": binding,
                    "seeded_violation_description": "unauthorized core dependency without declared owner",
                }
            )
        elif frz == "FRZ-DEBT-03":
            row["evidence_seeded_red_required"] = True
            node = ga2r["nodes"][0]
            node.update(
                {
                    "seeded_red": True,
                    "seeded_negative_node": binding,
                    "seeded_violation_description": "temporary compatibility seam without gate report",
                }
            )
        ga2r_by_frz[frz] = ga2r
    ref = "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json"
    with _patch_evidence_file(mod, {ref: ga2r_by_frz}):
        errors = mod.validate_pre_g2_op_b_settlement(plan["code_fix_inventory"], data=plan, git_validate=False)
    assert not errors


def test_pre_g2_does_not_set_rw_plan_audit_pass():
    mod = _load_verify()
    notes = mod.validate_rw_plan_audit(_live_rw_plan())
    assert notes
    assert "NOT_PERFORMED" not in " ".join(notes) or "independent" in notes[0].lower()
