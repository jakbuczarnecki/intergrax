# © Artur Czarnecki. All rights reserved.

"""Immutable multi-region security audit trail wiring (AUDIT-IDEAL-23.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.security.multi_region_audit_trail import (
    MultiRegionAuditTrailReport,
    MultiRegionSecurityAuditTrail,
)


@dataclass(frozen=True, slots=True)
class SecurityAuditTrailWiring:
    enabled: bool
    report: MultiRegionAuditTrailReport | None


def resolve_security_audit_trail_wiring(
    env: ApplicationEnvironmentProfile,
) -> SecurityAuditTrailWiring:
    """Replicate append-only security audit events across configured regions."""
    security = env.security_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return SecurityAuditTrailWiring(enabled=False, report=None)
    if not security.immutable_audit_trail_enabled:
        return SecurityAuditTrailWiring(enabled=False, report=None)
    if len(security.audit_trail_regions) < 2:
        return SecurityAuditTrailWiring(enabled=False, report=None)

    trail = MultiRegionSecurityAuditTrail(regions=tuple(sorted(security.audit_trail_regions)))
    trail.append(
        tenant_id=env.profile_id,
        action="audit_trail.bootstrap",
        actor_id="harness.bootstrap",
        resource="security.audit_trail",
        metadata={"regions": list(security.audit_trail_regions)},
    )
    trail.seal_prefix()
    trail.append(
        tenant_id=env.profile_id,
        action="audit_trail.ready",
        actor_id="harness.bootstrap",
        resource="security.audit_trail",
    )
    report = trail.verify_replication()
    if not report.replicated:
        return SecurityAuditTrailWiring(enabled=False, report=None)
    return SecurityAuditTrailWiring(enabled=True, report=report)
