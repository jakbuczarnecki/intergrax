from __future__ import annotations

from pathlib import Path

from intergrax.platform.enterprise_audit.contracts import (
    AuditCategory,
    AuditFinding,
    AuditMode,
    AuditSeverity,
    DriftStatus,
    EnterpriseAuditExitCode,
)
from intergrax.platform.enterprise_audit.manifest import checks_for_mode, load_default_manifest
from intergrax.platform.enterprise_audit.modes import aggregate_status, compute_exit_code


def test_manifest_has_eighty_five_checks() -> None:
    checks = load_default_manifest()
    assert len(checks) == 85


def test_fast_mode_skips_nothing_when_all_executable() -> None:
    checks = load_default_manifest()
    fast = checks_for_mode(checks, AuditMode.FAST)
    assert all(c.entrypoint_kind != "internal_pending" for c in fast)
    assert len(fast) == len(checks)


def test_exit_code_clean_when_no_findings() -> None:
    status = aggregate_status((), incomplete=False, internal_error=False)
    code = compute_exit_code(status, ())
    assert code is EnterpriseAuditExitCode.CLEAN


def test_resolved_drift_does_not_block_aggregate() -> None:
    resolved_blocker = AuditFinding(
        finding_id="OLD:1",
        check_id="ENT-A-BND-01",
        severity=AuditSeverity.BLOCKER,
        category=AuditCategory.ARCHITECTURE,
        message="Resolved since baseline",
        drift=DriftStatus.RESOLVED,
    )
    status = aggregate_status((resolved_blocker,), incomplete=False, internal_error=False)
    assert status.value == "PASS"


def test_manifest_json_matches_annex_g2_row_count() -> None:
    root = Path(__file__).resolve().parents[4]
    g2 = root / "docs/project/maintainers/qualification/ent_audit_x/ANNEX_G2_CONTROL_GATE_MAPPING.md"
    rows = sum(
        1
        for line in g2.read_text(encoding="utf-8").splitlines()
        if line.startswith("| ENT-") or line.startswith("| TEN-") or line.startswith("| VBN-")
    )
    assert rows == 85
