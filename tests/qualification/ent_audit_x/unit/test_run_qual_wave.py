from __future__ import annotations

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
