from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditMode,
    AuditSeverity,
    ManifestCheck,
)
from intergrax.platform.enterprise_audit.modes import (
    aggregate_status,
    relax_findings_for_non_certifying_fast,
)


def test_non_certifying_fast_downgrades_optional_gate_errors() -> None:
    checks = {
        "VBN-REACHABILITY": ManifestCheck(
            check_id="VBN-REACHABILITY",
            pg_id="PG-018",
            entrypoint="tests/x.py",
            entrypoint_kind="pytest_file",
            modes=(AuditMode.FAST,),
            severity_default=AuditSeverity.ERROR,
            category=AuditCategory.VBN,
        ),
    }
    findings = (
        AuditFinding(
            finding_id="VBN-REACHABILITY:FAIL",
            check_id="VBN-REACHABILITY",
            severity=AuditSeverity.ERROR,
            category=AuditCategory.VBN,
            message="fail",
        ),
    )
    relaxed = relax_findings_for_non_certifying_fast(
        AuditMode.FAST,
        certifying_run=False,
        findings=findings,
        checks_by_id=checks,
    )
    assert relaxed[0].severity is AuditSeverity.TRACKED_DEBT
    status = aggregate_status(relaxed, incomplete=False, internal_error=False)
    assert status.value == "PARTIAL"


def test_non_certifying_fast_keeps_mandatory_failures_blocking() -> None:
    checks = {
        "ENT-A-BND-01": ManifestCheck(
            check_id="ENT-A-BND-01",
            pg_id="PG-010",
            entrypoint="tests/y.py",
            entrypoint_kind="pytest_file",
            modes=(AuditMode.FAST,),
            fast_mandatory=True,
            severity_default=AuditSeverity.ERROR,
            category=AuditCategory.ARCHITECTURE,
        ),
    }
    findings = (
        AuditFinding(
            finding_id="ENT-A-BND-01:FAIL",
            check_id="ENT-A-BND-01",
            severity=AuditSeverity.ERROR,
            category=AuditCategory.ARCHITECTURE,
            message="fail",
        ),
    )
    relaxed = relax_findings_for_non_certifying_fast(
        AuditMode.FAST,
        certifying_run=False,
        findings=findings,
        checks_by_id=checks,
    )
    assert relaxed[0].severity is AuditSeverity.ERROR
    status = aggregate_status(relaxed, incomplete=False, internal_error=False)
    assert status.value == "BLOCKED"
