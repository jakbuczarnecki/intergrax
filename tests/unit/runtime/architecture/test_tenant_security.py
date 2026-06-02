from __future__ import annotations

from intergrax.runtime.architecture.tenant_security import (
    SecurityAuditEvent,
    TenantIsolationCheck,
    verify_tenant_security,
)


def test_tenant_security_fails_on_cross_tenant_access() -> None:
    report = verify_tenant_security(
        checks=[
            TenantIsolationCheck(
                request_tenant_id="tenant-a",
                resource_tenant_id="tenant-b",
                passed=False,
                reason="cross tenant",
            )
        ],
        audit_events=[
            SecurityAuditEvent(
                event_id="audit-1",
                tenant_id="tenant-a",
                actor_id="runtime",
                action="policy.deny",
            )
        ],
    )
    assert report.passed is False
    assert any("Tenant isolation check failed" in reason for reason in report.reasons)


def test_tenant_security_fails_when_audit_events_missing() -> None:
    report = verify_tenant_security(
        checks=[
            TenantIsolationCheck(
                request_tenant_id="tenant-a",
                resource_tenant_id="tenant-a",
                passed=True,
            )
        ],
        audit_events=[],
    )
    assert report.passed is False
    assert any("Missing security audit events" in reason for reason in report.reasons)
