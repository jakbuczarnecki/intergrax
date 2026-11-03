from __future__ import annotations

from intergrax.platform.enterprise_audit.baseline import EnterpriseAuditBaseline, apply_drift
from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditSeverity,
    DriftStatus,
    ManifestCheck,
)


def test_drift_regressed_severity() -> None:
    baseline = EnterpriseAuditBaseline(
        accepted_baseline_sha="abc",
        findings={
            "X:FAIL": {
                "check_id": "ENT-A-BND-01",
                "severity": AuditSeverity.WARNING.value,
                "message_fingerprint": "deadbeefdeadbeef",
            }
        },
    )
    finding = AuditFinding(
        finding_id="X:FAIL",
        check_id="ENT-A-BND-01",
        severity=AuditSeverity.ERROR,
        category=AuditCategory.ARCHITECTURE,
        message="worse",
    )
    drifted = apply_drift((finding,), baseline)
    assert drifted[0].drift is DriftStatus.REGRESSED


def test_drift_regressed_on_schema_bump() -> None:
    baseline = EnterpriseAuditBaseline(
        accepted_baseline_sha="abc",
        findings={
            "X:1": {
                "check_id": "ENT-A-BND-01",
                "severity": AuditSeverity.WARNING.value,
                "message_fingerprint": "deadbeefdeadbeef",
                "check_schema_version": 1,
            }
        },
    )
    check = ManifestCheck(
        check_id="ENT-A-BND-01",
        entrypoint_kind="pytest_nodeid",
        entrypoint="tests/unit/x.py::test_x",
        schema_version=2,
    )
    finding = AuditFinding(
        finding_id="X:1",
        check_id="ENT-A-BND-01",
        severity=AuditSeverity.WARNING,
        category=AuditCategory.ARCHITECTURE,
        message="same",
    )
    drifted = apply_drift((finding,), baseline, checks_by_id={check.check_id: check})
    assert drifted[0].drift is DriftStatus.REGRESSED


def test_drift_resolved_when_finding_absent() -> None:
    baseline = EnterpriseAuditBaseline(
        accepted_baseline_sha="abc",
        findings={
            "OLD:1": {
                "check_id": "ENT-A-BND-01",
                "severity": AuditSeverity.WARNING.value,
                "message_fingerprint": "abc",
            }
        },
    )
    drifted = apply_drift((), baseline)
    resolved = [f for f in drifted if f.drift is DriftStatus.RESOLVED]
    assert len(resolved) == 1
    assert resolved[0].finding_id == "OLD:1"
