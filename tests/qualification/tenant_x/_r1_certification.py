# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 qualification report and closure gates."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._authority_semantics import (
    TenantAuthorityReviewMetrics,
    build_authority_review_metrics,
)
from tests.qualification.tenant_x._r1_authority_resolutions import (
    SANCTIONED_ROOT_TENANT_SELECTORS,
    build_r1_authority_resolutions,
    build_r1_overlay_mechanisms,
    r1_duplicate_authority_confirmed,
    r1_duplicate_authority_unresolved,
)
from tests.qualification.tenant_x._r1_critical_boundaries import (
    R1_CRITICAL_BOUNDARY_FAMILIES,
    r1_critical_boundary_missing_or_invalid,
)
from tests.qualification.tenant_x._r1_propagation_proofs import (
    certify_r1_propagation_edges,
    r1_critical_propagation_missing,
)
from tests.qualification.tenant_x._r1_silent_substitution import (
    R1_SILENT_SUBSTITUTION_MATRIX,
    r1_critical_unknown_substitutions,
    r1_illegal_critical_substitutions,
)
from tests.qualification.tenant_x._r1_types import R1ProvenanceStatus
from tests.qualification.tenant_x._types import (
    TenantAuthorityOwnerDisposition,
    TenantAuthorityScopeEvidence,
    TenantSurfaceRecord,
)

LEGAL_CROSS_TENANT_PRODUCTION_MECHANISMS: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TenantXR1QualificationReport:
    critical_boundary_families: tuple[str, ...]
    propagation_edges: tuple[str, ...]
    authority_mechanisms_total: int
    scope_resolved: int
    scope_unresolved: int
    owner_confirmed: int
    owner_unresolved: int
    owner_conflict: int
    duplicate_confirmed: int
    duplicate_unresolved: int
    root_tenant_selectors: tuple[str, ...]
    legal_cross_tenant_mechanisms: int
    critical_unknown_substitutions: int
    illegal_critical_substitutions: int
    proven_current_head_edges: int
    historical_replay_edges: int
    missing_edges: int
    invalid_edges: int
    b2b_nordea_verdict: str
    r1_ready_for_audit: bool
    remediation_children: tuple[str, ...]


def _scope_unresolved(resolutions: tuple) -> int:
    return sum(
        1
        for r in resolutions
        if r.scope_evidence
        in {TenantAuthorityScopeEvidence.DOMAIN_FALLBACK, TenantAuthorityScopeEvidence.UNKNOWN}
    )


def build_r1_qualification_report(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> TenantXR1QualificationReport:
    resolutions = build_r1_authority_resolutions(inventory)
    overlay = build_r1_overlay_mechanisms(inventory)
    edges = certify_r1_propagation_edges()

    proven = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.PROVEN_CURRENT_HEAD)
    historical = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED)
    missing = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.MISSING)
    invalid = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.INVALID)

    owner_confirmed = sum(
        1 for r in resolutions if r.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
    )
    owner_unresolved = sum(
        1 for r in resolutions if r.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED
    )
    owner_conflict = sum(
        1 for r in resolutions if r.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFLICT
    )
    scope_unresolved = _scope_unresolved(resolutions)
    scope_resolved = len(resolutions) - scope_unresolved

    dup_confirmed = len(r1_duplicate_authority_confirmed(overlay))
    dup_unresolved = len(r1_duplicate_authority_unresolved(overlay))

    critical_unknown = len(r1_critical_unknown_substitutions())
    illegal_critical = len(r1_illegal_critical_substitutions())

    remediation: list[str] = []
    if r1_critical_boundary_missing_or_invalid():
        remediation.append("TENANT-X-R1-R1")
    if scope_unresolved or owner_unresolved or owner_conflict:
        remediation.append("TENANT-X-R1-R2")
    if dup_confirmed or dup_unresolved:
        remediation.append("TENANT-X-R1-R3")
    if r1_critical_propagation_missing():
        remediation.append("TENANT-X-R1-R4")
    if critical_unknown or illegal_critical:
        remediation.append("TENANT-X-R1-R5")

    ready = (
        not r1_critical_boundary_missing_or_invalid()
        and scope_unresolved == 0
        and owner_unresolved == 0
        and owner_conflict == 0
        and dup_confirmed == 0
        and dup_unresolved == 0
        and not r1_critical_propagation_missing()
        and critical_unknown == 0
        and illegal_critical == 0
        and len(resolutions) == 22
    )

    return TenantXR1QualificationReport(
        critical_boundary_families=tuple(f.family_id for f in R1_CRITICAL_BOUNDARY_FAMILIES),
        propagation_edges=tuple(e.edge_id for e in edges),
        authority_mechanisms_total=len(resolutions),
        scope_resolved=scope_resolved,
        scope_unresolved=scope_unresolved,
        owner_confirmed=owner_confirmed,
        owner_unresolved=owner_unresolved,
        owner_conflict=owner_conflict,
        duplicate_confirmed=dup_confirmed,
        duplicate_unresolved=dup_unresolved,
        root_tenant_selectors=SANCTIONED_ROOT_TENANT_SELECTORS,
        legal_cross_tenant_mechanisms=len(LEGAL_CROSS_TENANT_PRODUCTION_MECHANISMS),
        critical_unknown_substitutions=critical_unknown,
        illegal_critical_substitutions=illegal_critical,
        proven_current_head_edges=proven,
        historical_replay_edges=historical,
        missing_edges=missing,
        invalid_edges=invalid,
        b2b_nordea_verdict="READY WITH PLATFORM GAPS",
        r1_ready_for_audit=ready,
        remediation_children=tuple(sorted(set(remediation))),
    )


def r1_closure_passes(report: TenantXR1QualificationReport | None = None) -> bool:
    rep = report if report is not None else build_r1_qualification_report()
    return rep.r1_ready_for_audit


def summarize_r1_metrics(report: TenantXR1QualificationReport) -> dict[str, int | str | tuple[str, ...]]:
    return {
        "critical_provenance_proven_current_head": report.proven_current_head_edges,
        "critical_provenance_historical_replay": report.historical_replay_edges,
        "critical_provenance_missing": report.missing_edges,
        "critical_provenance_invalid": report.invalid_edges,
        "authority_mechanisms_total": report.authority_mechanisms_total,
        "scope_resolved": report.scope_resolved,
        "scope_unresolved": report.scope_unresolved,
        "owner_confirmed": report.owner_confirmed,
        "owner_unresolved": report.owner_unresolved,
        "owner_conflict": report.owner_conflict,
        "duplicate_authority_confirmed": report.duplicate_confirmed,
        "duplicate_authority_unresolved": report.duplicate_unresolved,
        "root_tenant_selectors": report.root_tenant_selectors,
        "legal_cross_tenant_mechanisms": report.legal_cross_tenant_mechanisms,
        "critical_unknown_substitutions": report.critical_unknown_substitutions,
        "silent_substitution_rows": len(R1_SILENT_SUBSTITUTION_MATRIX),
        "b2b_nordea": report.b2b_nordea_verdict,
        "remediation_children": report.remediation_children,
    }


def p0_authority_metrics_for_contrast(
    inventory: tuple[TenantSurfaceRecord, ...] | None = None,
) -> TenantAuthorityReviewMetrics:
    from tests.qualification.tenant_x._inventory import get_tenant_x_inventory

    inv = inventory if inventory is not None else get_tenant_x_inventory()
    return build_authority_review_metrics(inv)
