# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-I Part D: FAST promotion is disclosed and not applied to an incomplete floor (EQOA-A6-008). D01..D06."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.platform.enterprise_audit import runner as runner_module
from intergrax.platform.enterprise_audit.contracts import (
    AggregateAuditStatus,
    AuditCategory,
    AuditFinding,
    AuditMode,
    AuditSeverity,
    CheckRunOutcome,
    EnterpriseAuditExitCode,
    ManifestCheck,
)
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner
from intergrax.platform.enterprise_audit.validators import validate_report_semantic_consistency

_ROOT = Path(__file__).resolve().parents[4]
_OPTIONAL = "ENT-A-REL-02"
_MANDATORY = "ENT-A-BND-01"


def _check(check_id: str, *, mandatory: bool) -> ManifestCheck:
    return ManifestCheck(
        check_id=check_id,
        pg_id="PG-001",
        entrypoint="tests/x.py",
        entrypoint_kind="pytest_file",
        modes=frozenset(AuditMode),
        fast_mandatory=mandatory,
        severity_default=AuditSeverity.ERROR,
        category=AuditCategory.ARCHITECTURE,
    )


def _finding(check_id: str, severity: AuditSeverity) -> AuditFinding:
    return AuditFinding(
        finding_id=f"{check_id}:FAIL",
        check_id=check_id,
        severity=severity,
        category=AuditCategory.ARCHITECTURE,
        message="synthetic",
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mode: AuditMode = AuditMode.FAST,
    certifying: bool = False,
    floor_executed: bool = True,
    optional_severity: AuditSeverity = AuditSeverity.ERROR,
    mandatory_severity: AuditSeverity | None = None,
):
    checks = (_check(_MANDATORY, mandatory=True), _check(_OPTIONAL, mandatory=False))

    def fake_execute(check: ManifestCheck, _sandbox: object, *, mode: AuditMode) -> CheckRunOutcome:
        if check.check_id == _MANDATORY:
            if not floor_executed:
                return CheckRunOutcome(check_id=check.check_id, executed=False, skipped=True, skip_reason="synthetic")
            findings = (_finding(check.check_id, mandatory_severity),) if mandatory_severity else ()
            return CheckRunOutcome(
                check_id=check.check_id, executed=True, skipped=False, exit_code=1 if findings else 0, findings=findings
            )
        findings = (_finding(check.check_id, optional_severity),)
        return CheckRunOutcome(check_id=check.check_id, executed=True, skipped=False, exit_code=1, findings=findings)

    monkeypatch.setattr(runner_module, "execute_manifest_check", fake_execute)
    out = tmp_path / "run"
    runner = EnterpriseAuditRunner(_ROOT, manifest_checks=checks)
    changed = frozenset({_MANDATORY, _OPTIONAL}) if mode is AuditMode.CHANGED_SINCE else None
    result = runner.run(mode, output_dir=out, certifying_run=certifying, changed_check_ids=changed)
    return result, out


def test_wd_d01_promoted_run_discloses_status_and_debt_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (report, _manifest, code), _out = _run(monkeypatch, tmp_path)
    assert report.status is AggregateAuditStatus.PASS
    assert code is EnterpriseAuditExitCode.CLEAN
    promoted = [line for line in report.limitations if "status_promoted_from=PARTIAL" in line]
    assert len(promoted) == 1
    assert "tracked_debt_count=1" in promoted[0]


def test_wd_d02_incomplete_floor_is_not_promoted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (report, _manifest, code), _out = _run(monkeypatch, tmp_path, floor_executed=False)
    assert report.status is AggregateAuditStatus.PARTIAL
    assert code is EnterpriseAuditExitCode.INCOMPLETE
    assert not any("status_promoted_from" in line for line in report.limitations)


@pytest.mark.parametrize("mode", [AuditMode.FULL, AuditMode.FREEZE, AuditMode.CHANGED_SINCE])
def test_wd_d03_other_modes_are_never_promoted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: AuditMode
) -> None:
    (report, _manifest, code), _out = _run(
        monkeypatch, tmp_path, mode=mode, optional_severity=AuditSeverity.TRACKED_DEBT
    )
    assert report.status is AggregateAuditStatus.PARTIAL
    assert code is EnterpriseAuditExitCode.INCOMPLETE
    assert not any("status_promoted_from" in line for line in report.limitations)


def test_wd_d04_certifying_fast_is_never_promoted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (report, _manifest, code), _out = _run(
        monkeypatch, tmp_path, certifying=True, optional_severity=AuditSeverity.TRACKED_DEBT
    )
    assert report.status is AggregateAuditStatus.PARTIAL
    assert code is EnterpriseAuditExitCode.INCOMPLETE
    assert not any("status_promoted_from" in line for line in report.limitations)


def test_wd_d05_mixed_severities_are_not_promoted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (report, _manifest, code), _out = _run(monkeypatch, tmp_path, mandatory_severity=AuditSeverity.ERROR)
    assert report.status is AggregateAuditStatus.BLOCKED
    assert code is EnterpriseAuditExitCode.VIOLATION
    assert not any("status_promoted_from" in line for line in report.limitations)


def test_wd_d06_promoted_report_passes_semantic_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (report, _manifest, _code), out = _run(monkeypatch, tmp_path)
    assert report.status is AggregateAuditStatus.PASS
    path = out / "enterprise_audit_report.json"
    assert validate_report_semantic_consistency(path) == ()
