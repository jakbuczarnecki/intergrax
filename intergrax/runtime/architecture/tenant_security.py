# © Artur Czarnecki. All rights reserved.

"""Tenant isolation and audit trail verification contracts (Phase V-SEC.4)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class TenantIsolationCheck(BaseModel):
    request_tenant_id: str
    resource_tenant_id: str
    passed: bool
    reason: str = ""


class SecurityAuditEvent(BaseModel):
    event_id: str
    tenant_id: str
    actor_id: str
    action: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TenantSecurityVerificationReport(BaseModel):
    schema_version: str = "1.0.0"
    isolation_checks: list[TenantIsolationCheck] = Field(default_factory=list)
    audit_events: list[SecurityAuditEvent] = Field(default_factory=list)
    passed: bool
    reasons: list[str] = Field(default_factory=list)


def verify_tenant_security(
    *,
    checks: list[TenantIsolationCheck],
    audit_events: list[SecurityAuditEvent],
) -> TenantSecurityVerificationReport:
    reasons: list[str] = []
    for check in checks:
        if not check.passed:
            reasons.append(
                "Tenant isolation check failed: "
                f"{check.request_tenant_id} -> {check.resource_tenant_id}"
            )
    if not audit_events:
        reasons.append("Missing security audit events")
    return TenantSecurityVerificationReport(
        isolation_checks=checks,
        audit_events=audit_events,
        passed=not reasons,
        reasons=reasons,
    )
