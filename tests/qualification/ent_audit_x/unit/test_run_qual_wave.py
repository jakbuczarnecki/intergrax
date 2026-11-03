from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import scripts.enterprise_audit.qual_wave_lib as qw


def _root() -> Path:
    return Path(__file__).resolve().parents[4]


def test_pre_qual_status_not_fully_qualified_when_l3_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qw, "git_head", lambda _r: "a" * 40)
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "pytest_collect_count", lambda *a, **k: 63)
    monkeypatch.setattr(
        qw,
        "run_pytest_strict",
        lambda *a, **k: (0, "ok", qw.PytestSummary(passed=k.get("expected_passed") or 1)),
    )
    report = qw.run_qualification_wave(
        _root(),
        profile="pre-qual",
        pinned_sha=None,
        l3_output=tmp_path / "l3",
        wave_output=tmp_path / "wave",
    )
    assert report.status == "PRE_QUAL_PASS"
    assert any(s.name == "l3_cli_full" and s.skipped for s in report.stages)


def test_certifying_fails_when_l3_cli_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "b" * 40
    monkeypatch.setattr(qw, "git_head", lambda _r: sha)
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "pytest_collect_count", lambda *a, **k: 63)
    monkeypatch.setattr(
        qw,
        "run_pytest_strict",
        lambda *a, **k: (0, "ok", qw.PytestSummary(passed=k.get("expected_passed") or 1)),
    )
    monkeypatch.setattr(qw, "run_cli_full", lambda *a, **k: (1, "cli failed"))
    report = qw.run_qualification_wave(
        _root(),
        profile="certifying",
        pinned_sha=sha,
        l3_output=tmp_path / "l3",
        wave_output=tmp_path / "wave",
    )
    assert report.status == "FAIL"
    assert "L3 CLI" in report.error


def test_certifying_fails_on_dirty_tree_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "c" * 40
    monkeypatch.setattr(qw, "git_head", lambda _r: sha)
    monkeypatch.setattr(qw, "git_clean", lambda _r: False)
    report = qw.run_qualification_wave(
        _root(),
        profile="certifying",
        pinned_sha=sha,
        l3_output=tmp_path / "l3",
        wave_output=tmp_path / "wave",
    )
    assert report.error == "working tree not clean (clean checkout required)"


def test_validate_certifying_pin_missing() -> None:
    ok, msg = qw.validate_certifying_pin(_root(), None, certifying=True)
    assert not ok
    assert "TEST_EXECUTION_SHA required" in msg


def test_validate_certifying_pin_stale() -> None:
    head = qw.git_head(_root())
    ok, msg = qw.validate_certifying_pin(_root(), "0" * 40, certifying=True)
    assert not ok
    assert head in msg or "HEAD" in msg


def test_parse_pytest_summary_skipped_fails_strict() -> None:
    summary = qw.parse_pytest_summary("10 passed, 2 skipped in 1.0s")
    assert summary.passed == 10
    assert summary.skipped == 2


def test_semantic_stage_fails_on_skipped_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pytest(
        root,
        *,
        marker=None,
        paths=None,
        env=None,
        timeout_s=None,
        expected_passed=None,
        allow_skipped=False,
    ):
        if marker == "ent_semantic_gate" and expected_passed == qw._ENT_A_SEMANTIC_COUNT:
            return 1, "skipped=1", qw.PytestSummary(passed=62, skipped=1)
        return 0, "ok", qw.PytestSummary(passed=expected_passed or 1)

    monkeypatch.setattr(qw, "git_head", lambda _r: "d" * 40)
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "pytest_collect_count", lambda *a, **k: 63)
    monkeypatch.setattr(qw, "run_pytest_strict", fake_pytest)
    report = qw.run_qualification_wave(
        _root(),
        profile="pre-qual",
        pinned_sha=None,
        l3_output=tmp_path / "l3",
        wave_output=tmp_path / "wave",
    )
    assert report.status == "FAIL"
    assert "semantic_acceptance" in report.error


def test_semantic_matrix_fails_on_84_of_85(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pytest(*args, **kwargs):
        if kwargs.get("expected_passed") == 85:
            return 1, "passed 84 != expected 85", qw.PytestSummary(passed=84)
        return 0, "ok", qw.PytestSummary(passed=kwargs.get("expected_passed") or 1)

    monkeypatch.setattr(qw, "git_head", lambda _r: "e" * 40)
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "pytest_collect_count", lambda *a, **k: 63)
    monkeypatch.setattr(qw, "run_pytest_strict", fake_pytest)
    report = qw.run_qualification_wave(
        _root(),
        profile="pre-qual",
        pinned_sha=None,
        l3_output=tmp_path / "l3",
        wave_output=tmp_path / "wave",
    )
    assert report.status == "FAIL"
    assert "semantic_matrix" in report.error


def test_certifying_fails_without_freeze_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "f" * 40
    monkeypatch.setattr(qw, "git_head", lambda _r: sha)
    monkeypatch.setattr(qw, "git_clean", lambda _r: True)
    monkeypatch.setattr(qw, "pytest_collect_count", lambda *a, **k: 63)
    monkeypatch.setattr(
        qw,
        "run_pytest_strict",
        lambda *a, **k: (0, "ok", qw.PytestSummary(passed=k.get("expected_passed") or 1)),
    )
    monkeypatch.setattr(qw, "run_cli_full", lambda *a, **k: (0, "ok"))
    monkeypatch.setattr(qw, "validate_full_artifacts", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(qw, "run_first_freeze_candidate", lambda *a, **k: (1, "no candidate"))
    report = qw.run_qualification_wave(
        _root(),
        profile="certifying",
        pinned_sha=sha,
        l3_output=tmp_path / "l3",
        wave_output=tmp_path / "wave",
    )
    assert report.status == "FAIL"
    assert "FREEZE" in report.error


def test_acceptance_tests_not_collected_as_ent_l2() -> None:
    root = _root()
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "tests/qualification/ent_audit_x/acceptance",
            "-m",
            "ent_l2",
            "--collect-only",
            "-q",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "no tests collected" in proc.stdout.lower() or proc.returncode == 5


def test_orchestrator_certifying_rejects_before_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Certifying CLI fails closed on dirty tree or pin before expensive stages."""
    root = _root()
    monkeypatch.delenv("TEST_EXECUTION_SHA", raising=False)
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/enterprise_audit/run_qual_wave.py"),
            "--profile",
            "certifying",
            "--log-dir",
            str(tmp_path / "log"),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "TEST_EXECUTION_SHA" in proc.stderr or "working tree not clean" in proc.stderr or "HEAD" in proc.stderr


def test_validate_freeze_baseline_outside_jail() -> None:
    root = _root()
    outside = root / ".tmp" / "not-jailed-baseline"
    ok, msg = qw.validate_qual_wave_freeze_baseline_dir(root, outside, expected_sha="a" * 40)
    assert not ok
    assert "ent_audit_x" in msg


def test_validate_freeze_baseline_stale_candidate(tmp_path: Path) -> None:
    from intergrax.platform.enterprise_audit.baseline import (
        EnterpriseAuditBaseline,
        save_candidate_baseline,
    )

    root = tmp_path / "repo"
    baseline_dir = root / "tests/qualification/ent_audit_x/baselines/qual_wave_runs/run-a"
    baseline_dir.mkdir(parents=True)
    save_candidate_baseline(
        qw.candidate_baseline_path(baseline_dir),
        EnterpriseAuditBaseline(accepted_baseline_sha="b" * 40, findings={}),
    )
    ok, msg = qw.validate_qual_wave_freeze_baseline_dir(root, baseline_dir, expected_sha="a" * 40)
    assert not ok
    assert "stale candidate" in msg


def test_validate_freeze_baseline_collision_same_sha(tmp_path: Path) -> None:
    from intergrax.platform.enterprise_audit.baseline import (
        EnterpriseAuditBaseline,
        save_candidate_baseline,
    )

    sha = "c" * 40
    root = tmp_path / "repo"
    baseline_dir = root / "tests/qualification/ent_audit_x/baselines/qual_wave_runs/run-b"
    baseline_dir.mkdir(parents=True)
    save_candidate_baseline(
        qw.candidate_baseline_path(baseline_dir),
        EnterpriseAuditBaseline(accepted_baseline_sha=sha, findings={}),
    )
    ok, msg = qw.validate_qual_wave_freeze_baseline_dir(root, baseline_dir, expected_sha=sha)
    assert not ok
    assert "collision" in msg


def test_run_first_freeze_candidate_surfaces_runner_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    jailed = root / "tests/qualification/ent_audit_x/baselines/qual_wave_runs/wave-x"
    monkeypatch.setattr(qw, "qual_wave_freeze_baseline_dir", lambda _root, _wave: jailed)
    monkeypatch.setattr(qw, "git_head", lambda _r: "d" * 40)

    class _BoomRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, *args, **kwargs):
            raise RuntimeError("simulated FREEZE failure")

    monkeypatch.setattr(qw, "EnterpriseAuditRunner", _BoomRunner)
    wave = tmp_path / "wave-artifacts"
    wave.mkdir()
    code, detail = qw.run_first_freeze_candidate(root, wave)
    assert code == 1
    assert "FREEZE exception" in detail
    assert not qw.candidate_baseline_path(jailed).is_file()


def test_orchestrator_writes_fail_report_on_unhandled_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root()
    log_dir = tmp_path / "qual-log"
    sha = qw.git_head(root)
    monkeypatch.setenv("TEST_EXECUTION_SHA", sha)
    import scripts.enterprise_audit.run_qual_wave as rqw

    def _boom(*args, **kwargs):
        raise RuntimeError("orchestrator boom")

    monkeypatch.setattr(rqw.qw, "run_qualification_wave", _boom)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qual_wave.py",
            "--profile",
            "certifying",
            "--log-dir",
            str(log_dir),
        ],
    )
    assert rqw.main() == 1
    report_path = log_dir / "qual_wave_report.json"
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "orchestrator boom" in payload["error"]
