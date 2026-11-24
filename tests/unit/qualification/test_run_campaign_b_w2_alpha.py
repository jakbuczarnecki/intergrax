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
        row = mod.run_case(case, phase="b1", cwd=_REPO, execution_sha="c" * 40, dry_run=True)
        assert row["owner"] == "PT-B01"
    for case in manifest["b2_cases"]:
        row = mod.run_case(case, phase="b2", cwd=_REPO, execution_sha="d" * 40, dry_run=True)
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
    row = mod.run_case(case, phase="b1", cwd=tmp_path, execution_sha="e" * 40)
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
        rows = [mod.run_case(c, phase="b2", cwd=_REPO, execution_sha="f" * 40) for c in cases]
    assert len(rows) == 2


def test_evidence_persisted_and_digests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_runner()
    out = tmp_path / "out"
    monkeypatch.setattr(mod, "_OUT_ROOT", out)
    case = {"case_id": "Z", "contract": "AB", "command": f"{sys.executable} -c print('hi')"}
    row = mod.run_case(case, phase="b2", cwd=_REPO, execution_sha="1" * 40)
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
    gate = mod.BaselineValidation(True, [], execution_sha="2" * 40, cwd=_REPO)
    row = {
        "case_id": "A",
        "contract": "AB",
        "owner": "PT-B02",
        "result": "FAIL",
        "exit_code": 1,
    }
    with patch.object(mod, "validate_b2_cwd", return_value=gate):
        with patch.object(mod, "run_case", return_value=row):
            rc = mod.main(["--phase", "b2"])
    assert rc == 1
