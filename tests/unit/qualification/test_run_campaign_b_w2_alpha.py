# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RUNNER = _REPO / "scripts/qualification/run_campaign_b_w2_alpha.py"
_MANIFEST = _REPO / "scripts/qualification/campaign_b_w2_manifest.v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_campaign_b_w2_alpha", _RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_b1_without_worktree_rejected():
    mod = _load_runner()
    gate = mod.validate_b1_baseline(None, "a" * 40)
    assert not gate.ok


def test_nonexistent_worktree_rejected(tmp_path: Path):
    mod = _load_runner()
    gate = mod.validate_b1_baseline(tmp_path / "nope", "a" * 40)
    assert not gate.ok


def test_wrong_historical_sha_rejected(tmp_path: Path):
    mod = _load_runner()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    gate = mod.validate_b1_baseline(tmp_path, "b" * 40)
    assert not gate.ok
    assert head != "b" * 40


def test_b2_wrong_historical_sha_rejected(tmp_path: Path):
    mod = _load_runner()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    gate = mod.validate_b2_baseline(tmp_path, "b" * 40)
    assert not gate.ok


def test_dirty_worktree_rejected(tmp_path: Path):
    mod = _load_runner()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "dirty.txt").write_text("x", encoding="utf-8")
    gate = mod.validate_b1_baseline(tmp_path, head)
    assert not gate.ok


def test_no_subprocess_after_baseline_fail():
    mod = _load_runner()
    with patch.object(mod, "run_case") as mock_run:
        rc = mod.main(["--phase", "b1"])
        mock_run.assert_not_called()
        assert rc == 1


def test_owner_by_phase():
    mod = _load_runner()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    for case in manifest["b1_cases"]:
        row = mod.run_case(
            case,
            phase="b1",
            cwd=_REPO,
            execution_sha="c" * 40,
            execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
            instance_id="t",
            dry_run=True,
        )
        assert row["owner"] == "PT-B01"
    for case in manifest["b2_cases"]:
        row = mod.run_case(
            case,
            phase="b2",
            cwd=_REPO,
            execution_sha="d" * 40,
            execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
            instance_id="t",
            dry_run=True,
        )
        assert row["owner"] == "PT-B02"


def test_interpreter_path_with_spaces(tmp_path: Path):
    mod = _load_runner()
    exe = tmp_path / "my python.exe"
    exe.write_text("", encoding="utf-8")
    argv = mod._parse_command("python scripts/governance/verify_rw_plan_phase_check3.py")
    with patch.object(sys, "executable", str(exe)):
        argv2 = mod._parse_command("python scripts/governance/verify_rw_plan_phase_check3.py")
    assert argv2[0] == str(exe)
    assert "scripts/governance/verify_rw_plan_phase_check3.py" in argv2[1]
    assert argv[1] == "scripts/governance/verify_rw_plan_phase_check3.py"


def test_missing_artifact_not_executable_at_baseline(tmp_path: Path):
    mod = _load_runner()
    case = {"case_id": "X", "contract": "AB", "command": "python scripts/missing_at_baseline.py"}
    row = mod.run_case(
        case,
        phase="b1",
        cwd=tmp_path,
        execution_sha="e" * 40,
        execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
        instance_id="t",
    )
    assert row["result"] == "NOT_EXECUTABLE_AT_BASELINE"


def test_failed_case_does_not_stop_runner():
    mod = _load_runner()
    cases = [
        {"case_id": "A", "contract": "AB", "command": "python -c exit(1)"},
        {"case_id": "B", "contract": "AB", "command": "python -c print(1)"},
    ]
    with patch.object(
        mod.subprocess,
        "run",
        side_effect=[
            subprocess.CompletedProcess(["x"], 1, "", ""),
            subprocess.CompletedProcess(["x"], 0, "ok", ""),
        ],
    ):
        rows = [
            mod.run_case(
                c,
                phase="b2",
                cwd=_REPO,
                execution_sha="f" * 40,
                execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
                instance_id="t",
            )
            for c in cases
        ]
    assert len(rows) == 2


def test_evidence_persisted_and_digests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_runner()
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "_OUT_ROOT", out)
    case = {"case_id": "Z", "contract": "AB", "command": f"{sys.executable} -c print('hi')"}
    row = mod.run_case(
        case,
        phase="b2",
        cwd=_REPO,
        execution_sha="1" * 40,
        execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
        instance_id="t",
    )
    stdout = _REPO / row["stdout_path"]
    assert stdout.exists()
    assert "hi" in stdout.read_text(encoding="utf-8")
    assert row["stdout_sha256"] == mod._sha256_file(stdout)


def test_duplicate_manifest_case_rejected():
    mod = _load_runner()
    dup = {"case_id": "R", "contract": "AB", "command": "python -c pass"}
    errs = mod._validate_manifest_cases([dup, dup.copy()])
    assert any("duplicate case_id" in e for e in errs)


def test_aggregate_incomplete_nonzero():
    mod = _load_runner()
    gate = mod.BaselineValidation(
        True,
        [],
        execution_sha="2" * 40,
        cwd=_REPO,
        execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
    )
    row = {
        "case_id": "A",
        "contract": "AB",
        "owner": "PT-B02",
        "result": "FAIL",
        "exit_code": 1,
    }
    with patch.object(mod, "validate_b2_baseline", return_value=gate):
        with patch.object(mod, "run_case", return_value=row):
            with patch.object(mod, "_post_execution_tree_valid", return_value=[]):
                rc = mod.main(["--phase", "b2", "--worktree", str(_REPO)])
    assert rc == 1


def test_b2_without_worktree_rejected():
    mod = _load_runner()
    gate = mod.validate_b2_baseline(None, "a" * 40)
    assert not gate.ok


def test_pytest_node_resolves_source_file(tmp_path: Path):
    mod = _load_runner()
    test_file = tmp_path / "tests" / "unit" / "x.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_f(): pass\n", encoding="utf-8")
    argv = mod._parse_command("python -m pytest tests/unit/x.py::test_f -q")
    paths = mod._command_input_artifacts(argv)
    assert paths == [Path("tests/unit/x.py")]
    ok, missing = mod._artifact_available(tmp_path, argv)
    assert ok
    assert missing is None


def test_pytest_node_preserved_in_subprocess():
    mod = _load_runner()
    node = (
        "tests/unit/architecture/test_ebh_2d_d_application_contract_surface_recertification.py"
        "::test_whole_surface_forbids_contract_to_implementation_edges"
    )
    argv = mod._parse_command(f"python -m pytest {node} -q --tb=line")
    assert node in argv


def test_missing_output_dir_does_not_reject_r03(tmp_path: Path):
    mod = _load_runner()
    script = tmp_path / "scripts" / "governance" / "arch_freeze_b1_closed_world_inventory.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")
    cmd = (
        "python scripts/governance/arch_freeze_b1_closed_world_inventory.py "
        "--report-only --output-dir .tmp/session/w2-recovery-01/b1/closed-world"
    )
    argv = mod._parse_command(cmd)
    ok, _ = mod._artifact_available(tmp_path, argv)
    assert ok


def test_head_diagnostic_not_classified_as_alpha():
    mod = _load_runner()
    gate = mod.validate_head_diagnostic(_REPO)
    assert gate.execution_context == mod.EXECUTION_CONTEXT_HEAD
    assert gate.execution_context != mod.EXECUTION_CONTEXT_HISTORICAL


def test_partial_coverage_cannot_claim_alpha_complete():
    mod = _load_runner()
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    rows = [{"result": "PASS"} for _ in manifest["b1_cases"]]
    summary = mod._coverage_summary("b1", manifest["b1_cases"], rows)
    assert summary["defined_cases_pass"] is True
    assert summary["campaign_alpha_complete"] is False
    assert summary["coverage_complete"] is False


def test_evidence_paths_separate_by_execution_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_runner()
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "_OUT_ROOT", out)
    case = {"case_id": "Z", "contract": "AB", "command": f"{sys.executable} -c pass"}
    mod.run_case(
        case,
        phase="b2",
        cwd=_REPO,
        execution_sha="a" * 40,
        execution_context=mod.EXECUTION_CONTEXT_HISTORICAL,
        instance_id="i1",
    )
    mod.run_case(
        case,
        phase="b2",
        cwd=_REPO,
        execution_sha="a" * 40,
        execution_context=mod.EXECUTION_CONTEXT_HEAD,
        instance_id="i1",
    )
    assert (out / "b2" / mod.EXECUTION_CONTEXT_HISTORICAL).exists()
    assert (out / "b2" / mod.EXECUTION_CONTEXT_HEAD).exists()


def test_historical_tree_mutation_invalidates_qualification(tmp_path: Path):
    mod = _load_runner()
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "mut.txt").write_text("dirty", encoding="utf-8")
    errs = mod._post_execution_tree_valid(tmp_path, sha)
    assert any("dirty" in e for e in errs)
