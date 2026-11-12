# © Artur Czarnecki. All rights reserved.

"""Immutable multi-region security audit trail wiring (AUDIT-IDEAL-23.1, PDC-D6-R020/R024)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.core.security.security_configuration import (
    AUDIT_REGIONS_INVALID,
    AUDIT_REPLICATION_FAILED,
    AUDIT_SINK_MISSING,
    SecurityConfigurationError,
)
from intergrax.runtime.security.multi_region_audit_trail import (
    MultiRegionAuditTrailReport,
    MultiRegionSecurityAuditTrail,
)
from intergrax.runtime.security.security_audit_trail import (
    EvidencePlaneSecurityAuditTrail,
    SecurityAuditSink,
)

_ITEM = "security_audit_trail"


@dataclass(frozen=True, slots=True)
class SecurityAuditTrailWiring:
    enabled: bool
    report: MultiRegionAuditTrailReport | None
    trail: SecurityAuditSink | None = None


def resolve_security_audit_trail_wiring(
    env: ApplicationEnvironmentProfile,
    *,
    evidence_persistence: EvidencePersistencePort | None = None,
) -> SecurityAuditTrailWiring:
    """Persist append-only security audit events through the canonical evidence plane.

    On a product host with ``immutable_audit_trail_enabled`` the evidence persistence port
    is mandatory and at least two regions must be configured; a missing sink or malformed
    region list raises :class:`SecurityConfigurationError` (no in-memory fallback).
    """
    security = env.security_profile
    if env.application_profile is not ApplicationProfile.PRODUCT:
        return SecurityAuditTrailWiring(enabled=False, report=None)
    if not security.immutable_audit_trail_enabled:
        return SecurityAuditTrailWiring(enabled=False, report=None)
    regions = tuple(sorted(r.strip() for r in security.audit_trail_regions if r and r.strip()))
    if len(set(regions)) < 2 or len(regions) != len(security.audit_trail_regions):
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=AUDIT_REGIONS_INVALID,
            message="audit_trail_regions must list at least two distinct non-blank regions",
        )
    if evidence_persistence is None:
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=AUDIT_SINK_MISSING,
            message="immutable audit trail requires an EvidencePersistencePort (no in-memory fallback)",
        )

    durable = EvidencePlaneSecurityAuditTrail(evidence_persistence)
    trail = MultiRegionSecurityAuditTrail(regions=regions, primary=durable)
    trail.append(
        tenant_id=env.profile_id,
        action="audit_trail.bootstrap",
        actor_id="harness.bootstrap",
        resource="security.audit_trail",
        metadata={"regions": list(regions)},
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
        raise SecurityConfigurationError(
            item=_ITEM,
            reason_code=AUDIT_REPLICATION_FAILED,
            message="audit trail replication check failed at bootstrap",
        )
    return SecurityAuditTrailWiring(enabled=True, report=report, trail=trail)
