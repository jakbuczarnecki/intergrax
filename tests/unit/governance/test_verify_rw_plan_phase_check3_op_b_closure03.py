# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

_REPO = Path(__file__).resolve().parents[3]
_VERIFY = _REPO / "scripts/governance/verify_rw_plan_phase_check3.py"
_RW_JSON = _REPO / "docs/project/maintainers/qualification/arch_freeze_program/rw_plan.v1.json"
_GA2R_REF = "docs/project/maintainers/qualification/arch_freeze_program/continuation/RED_FIRST_GA2R_RESULT.json"
_EVIDENCE_REL = f"{_GA2R_REF}#FRZ-BND-05"


def _load_verify():
    spec = importlib.util.spec_from_file_location("verify_rw_plan_phase_check3", _VERIFY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _live_rw_plan() -> dict:
    return json.loads(_RW_JSON.read_text(encoding="utf-8"))


def _complete_node(binding: str, *, prefix_sha: str = "a" * 40, pin_sha: str = "f" * 40) -> dict:
    return {
        "node": binding,
        "counted": True,
        "prefix_sha": prefix_sha,
        "fix_commit": "b" * 40,
        "pin_sha": pin_sha,
        "test_blob_sha": "c" * 40,
        "pin_blob_sha": "d" * 40,
        "prefix_result": "FAILED",
        "prefix_run_exit": 1,
        "pin_result": "PASSED",
        "pin_run_exit": 0,
        "assertion_lines": ["E       AssertionError: synthetic mechanism failure"],
        "prefix_log_sha256": "e" * 64,
        "pin_log_sha256": "0" * 64,
    }


def _complete_ga2r(frz: str, binding: str, prefix_sha: str = "a" * 40) -> dict:
    return {
        "frz_id": frz,
        "status": "RED_AT_PREFIX",
        "mechanism": "synthetic mechanism for progressive contract test",
        "nodes": [_complete_node(binding, prefix_sha=prefix_sha)],
    }


def _patch_evidence_file(mod, file_index: dict[str, dict]):
    def _fake_load(path: Path):
        rel = path.relative_to(_REPO).as_posix()
        if rel in file_index:
            by_frz = file_index[rel]
            return dict(by_frz), []
        return mod._load_ga2r_index_from_file(path)

    return patch.object(mod, "_load_ga2r_index_from_file", side_effect=_fake_load)


def _op_b_row(plan: dict, frz: str = "FRZ-BND-05") -> tuple[dict, str]:
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    binding = "tests/unit/architecture/test_rw_bnd05_runtime_context_construction.py::test_scan_flags_a_rogue_agent"
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": "a" * 40,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
            "evidence_negative_test": binding,
        }
    )
    return row, binding


def test_op_b_red_at_prefix_complete_record_passes():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    ga2r = _complete_ga2r(frz, binding)
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert not [e for e in errors if "OP-B RED_AT_PREFIX" in e]


def test_evidence_ref_file_a_cannot_use_records_from_file_b():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    alt = "docs/project/maintainers/qualification/arch_freeze_program/continuation/OTHER_EVIDENCE.json"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    row["red_first_evidence_ref"] = f"{alt}#{frz}"
    ga2r = _complete_ga2r(frz, binding)
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}, alt: {}}):
        with patch.object(mod, "_git_tracked", return_value=True):
            errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("without GA2R inventory row" in e for e in errors)


def test_untracked_evidence_file_rejected():
    mod = _load_verify()
    plan = _live_rw_plan()
    _op_b_row(plan)
    with patch.object(mod, "_git_tracked", return_value=False):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("git-tracked" in e for e in errors)


def test_path_traversal_evidence_ref_rejected():
    mod = _load_verify()
    plan = _live_rw_plan()
    row, _ = _op_b_row(plan)
    row["red_first_evidence_ref"] = (
        "docs/project/maintainers/qualification/arch_freeze_program/continuation/../"
        "RED_FIRST_GA2R_RESULT.json#FRZ-BND-05"
    )
    errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("evidence ref" in e or "authorized" in e for e in errors)


def test_invalid_json_rejected_without_unhandled_exception():
    mod = _load_verify()
    plan = _live_rw_plan()
    _op_b_row(plan)

    def _bad_json(path: Path):
        return {}, ["evidence file invalid JSON: Expecting value"]

    with patch.object(mod, "_load_ga2r_index_from_file", side_effect=_bad_json):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("invalid JSON" in e for e in errors)


def test_duplicate_frz_id_in_evidence_file_rejected():
    mod = _load_verify()
    plan = _live_rw_plan()
    _op_b_row(plan)

    def _dup_load(path: Path):
        return {}, ["evidence file duplicate frz_id FRZ-BND-05"]

    with patch.object(mod, "_load_ga2r_index_from_file", side_effect=_dup_load):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("duplicate frz_id" in e for e in errors)


def test_wrong_frz_anchor_rejected():
    mod = _load_verify()
    plan = _live_rw_plan()
    row, _ = _op_b_row(plan)
    row["red_first_evidence_ref"] = f"{_GA2R_REF}#FRZ-UNKNOWN"
    errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("anchor" in e or "FRZ" in e for e in errors)


def test_missing_pin_sha_rejected():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node.pop("pin_sha", None)
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("pin_sha" in e for e in errors)


def test_nonexistent_commit_rejected_when_git_validate():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    ga2r = _complete_ga2r(frz, binding, prefix_sha="a" * 40)
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        with patch.object(mod, "_git_object_exists", return_value=False):
            errors = mod.validate_red_first_evidence(plan, git_validate=True)
    assert any("not found in repository" in e for e in errors)


def test_invalid_ancestry_rejected():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    ga2r = _complete_ga2r(frz, binding)
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        with patch.object(mod, "_git_object_exists", return_value=True):
            with patch.object(mod, "_git_is_ancestor", return_value=False):
                errors = mod.validate_red_first_evidence(plan, git_validate=True)
    assert any("descend" in e or "contain" in e for e in errors)


def test_missing_exit_codes_rejected():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node.pop("prefix_run_exit")
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("exit" in e for e in errors)


def test_wrong_exit_codes_rejected():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node["pin_run_exit"] = 1
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("pin_run_exit=0" in e for e in errors)


def test_missing_test_version_digest_rejected():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node.pop("pin_blob_sha")
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("pin_blob_sha" in e for e in errors)


def test_missing_log_digest_rejected():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node.pop("pin_log_sha256")
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("green measurement log digest" in e for e in errors)


def test_collection_error_assertion_not_red_first():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node["assertion_lines"] = ["E   ModuleNotFoundError: fastmcp"]
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("mechanism-aligned assertion" in e for e in errors)


def test_e_prefix_only_assertion_insufficient():
    mod = _load_verify()
    frz = "FRZ-BND-05"
    plan = _live_rw_plan()
    row, binding = _op_b_row(plan, frz)
    node = _complete_node(binding)
    node["assertion_lines"] = ["E       some pytest output without assertion semantics"]
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("mechanism-aligned assertion" in e for e in errors)


def test_seeded_name_in_test_not_evidence():
    mod = _load_verify()
    frz = "FRZ-DEBT-02"
    binding = (
        "tests/unit/architecture/test_rw_deps_core_ownership_declared.py::"
        "test_rw_dep_01c_gate_reports_seeded_undeclared_core_requirement"
    )
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": "a" * 40,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
            "evidence_negative_test": binding,
            "evidence_seeded_red_required": True,
        }
    )
    ga2r = _complete_ga2r(frz, binding)
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("seeded-red" in e for e in errors)


def test_seeded_red_flag_alone_insufficient():
    mod = _load_verify()
    frz = "FRZ-DEBT-02"
    binding = "tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01b"
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": "a" * 40,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
            "evidence_negative_test": binding,
            "evidence_seeded_red_required": True,
        }
    )
    node = _complete_node(binding)
    node["seeded_red"] = True
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("seeded_negative_node" in e or "seeded_violation" in e for e in errors)


def test_complete_seeded_red_record_meets_mechanical_contract():
    mod = _load_verify()
    frz = "FRZ-DEBT-02"
    binding = "tests/unit/architecture/test_rw_deps_core_ownership_declared.py::test_rw_dep_01b"
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == frz)
    row.update(
        {
            "red_first_status": "RED_AT_PREFIX",
            "red_first_prefix_sha": "a" * 40,
            "red_first_evidence_ref": f"{_GA2R_REF}#{frz}",
            "evidence_negative_test": binding,
            "evidence_seeded_red_required": True,
        }
    )
    node = _complete_node(binding)
    node.update(
        {
            "seeded_red": True,
            "seeded_negative_node": binding,
            "seeded_violation_description": "unauthorized core dependency without declared owner",
        }
    )
    ga2r = {"frz_id": frz, "nodes": [node]}
    with _patch_evidence_file(mod, {_GA2R_REF: {frz: ga2r}}):
        errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert not [e for e in errors if "seeded" in e.lower() and "requires" in e.lower()]


def test_structural_pass_does_not_set_rw_plan_audit_pass():
    mod = _load_verify()
    audit = mod.validate_rw_plan_audit(_live_rw_plan())
    assert audit
    assert "not performed" in audit[0].lower()


def test_git_ancestry_with_temporary_repository(tmp_path: Path):
    mod = _load_verify()
    repo = tmp_path / "mini.git"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("1", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "c1"], cwd=repo, check=True)
    sha1 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "f.txt").write_text("2", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "c2"], cwd=repo, check=True)
    sha2 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "f.txt").write_text("3", encoding="utf-8")
    subprocess.run(["git", "commit", "-am", "c3"], cwd=repo, check=True)
    sha3 = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    node = {"prefix_sha": sha1, "fix_commit": sha2, "pin_sha": sha3}
    errors: list[str] = []
    with patch.object(mod, "_REPO", repo):
        mod._validate_op_b_execution_graph("FRZ-X", node, errors, git_validate=True)
    assert not errors


def test_invalid_prefix_sha_characters_rejected():
    mod = _load_verify()
    plan = _live_rw_plan()
    row = next(r for r in plan["code_fix_inventory"] if r["frz_id"] == "FRZ-BND-05")
    row["red_first_status"] = "RED_AT_PREFIX"
    row["red_first_prefix_sha"] = "g" * 40
    row["red_first_evidence_ref"] = _EVIDENCE_REL
    errors = mod.validate_red_first_evidence(plan, git_validate=False)
    assert any("valid historical prefix SHA" in e or "invalid characters" in e for e in errors)
