from __future__ import annotations

from intergrax.platform.enterprise_audit.contracts import (
    AggregateAuditStatus,
    AuditCategory,
    AuditFinding,
    AuditSeverity,
    DriftStatus,
)
from intergrax.platform.enterprise_audit.modes import aggregate_status


def test_resolved_finding_does_not_block_aggregate() -> None:
    resolved_blocker = AuditFinding(
        finding_id="OLD:1",
        check_id="ENT-A-BND-01",
        severity=AuditSeverity.BLOCKER,
        category=AuditCategory.ARCHITECTURE,
        message="was blocker",
        drift=DriftStatus.RESOLVED,
    )
    status = aggregate_status((resolved_blocker,), incomplete=False, internal_error=False)
    assert status is AggregateAuditStatus.PASS
