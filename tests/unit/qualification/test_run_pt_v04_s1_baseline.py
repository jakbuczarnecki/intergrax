# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_RUNNER = _REPO / "scripts/qualification/run_pt_v04_s1_baseline.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_pt_v04_s1_baseline", _RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_f10_only_check_argv():
    text = _RUNNER.read_text(encoding="utf-8")
    assert "render_ebh7_f10_normative_matrix.py" in text
    assert "--check" in text
    assert text.count("render_ebh7_f10_normative_matrix.py") == 1


def test_no_mutating_f10_render():
    text = _RUNNER.read_text(encoding="utf-8")
    assert "F10_render" not in text
    assert '"--check"' in text


def test_dirty_tree_rejected_at_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _load_runner()
    monkeypatch.setattr(mod, "_OUT", tmp_path)
    monkeypatch.setattr(mod, "_sha", lambda _cwd: "a" * 40)
    monkeypatch.setattr(mod, "_worktree_clean", lambda _cwd: False)
    rc = mod.main()
    assert rc == 1
    out = json.loads((tmp_path / "aaaaaaaaaaaa" / "PT_V04_S1_BASELINE.json").read_text(encoding="utf-8"))
    assert out["certifying"] is False


def test_runner_non_certifying_flag():
    text = _RUNNER.read_text(encoding="utf-8")
    assert '"certifying": False' in text


def test_evidence_files_and_digests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mod = _load_runner()
    monkeypatch.setattr(mod, "_OUT", tmp_path)
    step = mod._run_step([sys.executable, "-c", "print('x')"], "t", tmp_path / "ev")
    assert Path(_REPO / step["stdout_path"]).read_text(encoding="utf-8").strip() == "x"
    assert step["stdout_sha256"] == mod._sha256_file(_REPO / step["stdout_path"])


def test_failed_subprocess_incomplete(tmp_path: Path):
    mod = _load_runner()
    step = mod._run_step([sys.executable, "-c", "raise SystemExit(2)"], "fail", tmp_path)
    assert step["exit_code"] == 2
    assert not step["complete"]
