# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_VERIFY = _REPO / "scripts/governance/verify_rw_plan_phase_check3.py"
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
_GA2R_REF = "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json"


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_rw_plan_phase_check3", _VERIFY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _live_rw_plan() -> dict:
    return json.loads(_RW_JSON.read_text(encoding="utf-8"))


def _synthetic_complete_ga2r(frz: str, binding: str, prefix_sha: str) -> dict:
    return {
        "frz_id": frz,
        "status": "RED_AT_PREFIX",
        "mechanism": "synthetic mechanism for progressive contract test",
        "nodes": [
            {
                "node": binding,
                "counted": True,
                "prefix_sha": prefix_sha,
                "fix_commit": "b" * 40,
                "test_blob_sha": "c" * 40,
                "pin_blob_sha": "d" * 40,
                "prefix_result": "FAILED",
                "prefix_run_exit": 1,
                "pin_result": "PASSED",
                "pin_run_exit": 0,
                "assertion_lines": ["E       AssertionError: synthetic mechanism failure"],
                "prefix_log_sha256": "e" * 64,
            }
        ],
    }


def test_op_b_red_at_prefix_complete_record_passes():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    binding = "tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py::test_scan_flags_a_rogue_agent"
    prefix_sha = "a" * 40
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": prefix_sha,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
        }
    )
    ga2r_index = {frz: _synthetic_complete_ga2r(frz, binding, prefix_sha)}
    errors = mod.validate_red_first_evidence(plan, ga2r_index=ga2r_index)
    assert not [e for e in errors if "OP-B RED_AT_PREFIX" in e]


@pytest.mark.parametrize(
    "mutator,fragment",
    [
        (lambda r: r.update({"red_first_prefix_sha": "Z" * 40}), "invalid characters"),
        (lambda r: r.update({"red_first_evidence_ref": "docs/missing.json#FRZ-BND-05"}), "existing tracked record"),
        (
            lambda r: r.update({"red_first_evidence_ref": f"{_GA2R_REF}#FRZ-UNKNOWN"}),
            "for this FRZ",
        ),
        (lambda r: r.update({"evidence_negative_test": "tests/unit/no/such.py::test_x"}), "missing GA2R node"),
    ],
)
def test_op_b_red_at_prefix_incomplete_rejected(mutator, fragment: str):
    mod = _load_verify()
    frz = "FRZ-BND-05"
    binding = "tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py::test_scan_flags_a_rogue_agent"
    prefix_sha = "a" * 40
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": prefix_sha,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
        }
    )
    mutator(row)
    ga2r_index = {frz: _synthetic_complete_ga2r(frz, binding, prefix_sha)}
    errors = mod.validate_red_first_evidence(plan, ga2r_index=ga2r_index)
    assert any(fragment in e for e in errors)


def test_op_b_collection_error_assertion_not_red_first():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    binding = "tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py::test_scan_flags_a_rogue_agent"
    prefix_sha = "a" * 40
    record = _synthetic_complete_ga2r(frz, binding, prefix_sha)
    record["nodes"][0]["assertion_lines"] = ["E   ModuleNotFoundError: fastmcp"]
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": prefix_sha,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
        }
    )
    errors = mod.validate_red_first_evidence(plan, ga2r_index={frz: record})
    assert any("mechanism-aligned assertion" in e for e in errors)


def test_op_b_seeded_red_promotion_requires_seeded_binding():
    mod = _load_verify()
    frz = "FRZ-DEBT-02"
    binding = (
        "tests/unit/architecture/test_rw_deps_core_ownership_declared.py::"
        "test_rw_dep_01a_core_requirement_has_declared_owner"
    )
    prefix_sha = "a" * 40
    record = _synthetic_complete_ga2r(frz, binding, prefix_sha)
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": prefix_sha,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
            "evidence_negative_test": binding,
        }
    )
    errors = mod.validate_red_first_evidence(plan, ga2r_index={frz: record})
    assert any("seeded-red" in e for e in errors)


def test_invalid_prefix_sha_characters_rejected():
    mod = _load_verify()
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-BND-05")
    row["red_first_status"] = "RED_AT_PREFIX"
    row["red_first_prefix_sha"] = "g" * 40
    row["red_first_evidence_ref"] = f"{_GA2R_REF}#FRZ-BND-05"
    errors = mod.validate_red_first_evidence(plan, ga2r_index={})
    assert any("valid historical prefix SHA" in e or "invalid characters" in e for e in errors)
