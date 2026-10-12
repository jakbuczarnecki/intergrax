# © Artur Czarnecki. All rights reserved.

"""TENANT-X-R1 qualification report and closure gates."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._authority_semantics import (
    TenantAuthorityReviewMetrics,
    build_authority_review_metrics,
)
from tests.qualification.tenant_x._r1_authority_resolutions import (
    build_r1_authority_resolutions,
    build_r1_overlay_mechanisms,
    build_r1_scoped_duplicate_authority_review,
    build_sanctioned_root_tenant_selectors,
    r1_critical_owner_unresolved,
    r1_critical_scope_unresolved,
    r1_duplicate_authority_confirmed,
    r1_duplicate_authority_unresolved,
)
from tests.qualification.tenant_x._r1_capability_routing_consumers import (
    capability_routing_consumer_unknown_count,
    routing_sentinel_direct_authority_escape_paths,
)
from tests.qualification.tenant_x._r1_cross_tenant_inventory import (
    build_r1_cross_tenant_inventory,
    legal_cross_tenant_mechanism_count,
    r1_cross_tenant_unknown_critical,
)
from tests.qualification.tenant_x._r1_critical_boundaries import (
    R1_CRITICAL_BOUNDARY_FAMILIES,
    r1_owned_critical_boundary_missing_or_invalid,
)
from tests.qualification.tenant_x._r1_propagation_proofs import (
    certify_r1_propagation_edges,
    r1_owned_propagation_missing,
)
from tests.qualification.tenant_x._r1_root_selectors import (
    build_r1_root_selector_inventory,
    r1_critical_unknown_root_selectors,
    r1_illegal_root_selectors,
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
    critical_authority_participants: int
    non_authority_duplicate_participants: int
    unknown_authority_participation: int
    non_critical_owner_scope_uncertainty: int
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
    inv = inventory if inventory is not None else None
    resolutions = build_r1_authority_resolutions(inv)
    overlay = build_r1_overlay_mechanisms(inv)
    scoped_duplicate = build_r1_scoped_duplicate_authority_review(inv)
    edges = certify_r1_propagation_edges()
    root_inventory = build_r1_root_selector_inventory(inventory)
    root_selectors = build_sanctioned_root_tenant_selectors(inventory)

    proven = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.PROVEN_CURRENT_HEAD)
    historical = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.HISTORICAL_REPLAY_CONFIRMED)
    missing = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.MISSING)
    invalid = sum(1 for e in edges if e.final_status == R1ProvenanceStatus.INVALID)

    owner_confirmed = sum(
        1 for r in resolutions if r.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
    )
    owner_unresolved = sum(
        1
        for r in resolutions
        if r.owner_disposition
        in {
            TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED,
            TenantAuthorityOwnerDisposition.OWNER_CANDIDATE,
        }
    )
    owner_conflict = sum(
        1 for r in resolutions if r.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFLICT
    )
    scope_unresolved = _scope_unresolved(resolutions)
    scope_resolved = len(resolutions) - scope_unresolved

    dup_confirmed = len(r1_duplicate_authority_confirmed(overlay, inv))
    dup_unresolved = len(r1_duplicate_authority_unresolved(overlay, inv))

    critical_unknown = len(r1_critical_unknown_substitutions())
    illegal_critical = len(r1_illegal_critical_substitutions())

    remediation: list[str] = []
    if r1_owned_critical_boundary_missing_or_invalid():
        remediation.append("TENANT-X-R1-R1 - R1-owned critical boundary families missing/invalid")
    if r1_critical_scope_unresolved(resolutions) or r1_critical_owner_unresolved(resolutions):
        remediation.append("TENANT-X-R1-R2 - critical authority scope/owner reconciliation")
    if (
        dup_confirmed
        or dup_unresolved
        or scoped_duplicate.unknown_participation_ids
    ):
        remediation.append("TENANT-X-R1-R3 - duplicate authority reconciliation")
    if r1_owned_propagation_missing():
        remediation.append("TENANT-X-R1-R4 - missing R1-owned structural provenance")
    if critical_unknown or illegal_critical:
        remediation.append("TENANT-X-R1-R5 - silent substitution gaps")
    if len(root_selectors) != 1 or r1_illegal_root_selectors(root_inventory):
        remediation.append("TENANT-X-R1-R6 - root tenant selector inventory")
    if r1_critical_unknown_root_selectors(root_inventory):
        remediation.append("TENANT-X-R1-R7 - unknown critical root selectors")
    if capability_routing_consumer_unknown_count() or routing_sentinel_direct_authority_escape_paths():
        remediation.append("TENANT-X-R1-R3-Q1-R1 - capability routing sentinel reconciliation")

    ready = (
        not r1_owned_critical_boundary_missing_or_invalid()
        and not r1_critical_scope_unresolved(resolutions)
        and not r1_critical_owner_unresolved(resolutions)
        and owner_conflict == 0
        and dup_confirmed == 0
        and dup_unresolved == 0
        and not scoped_duplicate.unknown_participation_ids
        and not r1_owned_propagation_missing()
        and critical_unknown == 0
        and illegal_critical == 0
        and len(root_selectors) == 1
        and not r1_illegal_root_selectors(root_inventory)
        and not r1_critical_unknown_root_selectors(root_inventory)
        and not r1_cross_tenant_unknown_critical(build_r1_cross_tenant_inventory(inventory))
        and len(resolutions) == 22
        and capability_routing_consumer_unknown_count() == 0
        and not routing_sentinel_direct_authority_escape_paths()
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
        critical_authority_participants=len(scoped_duplicate.critical_authority_participant_ids),
        non_authority_duplicate_participants=len(scoped_duplicate.non_authority_participant_ids),
        unknown_authority_participation=len(scoped_duplicate.unknown_participation_ids),
        non_critical_owner_scope_uncertainty=len(
            scoped_duplicate.non_critical_owner_scope_uncertainty_ids
        ),
        root_tenant_selectors=root_selectors,
        legal_cross_tenant_mechanisms=legal_cross_tenant_mechanism_count(
            build_r1_cross_tenant_inventory(inventory)
        ),
        critical_unknown_substitutions=critical_unknown,
        illegal_critical_substitutions=illegal_critical,
        proven_current_head_edges=proven,
        historical_replay_edges=historical,
        missing_edges=missing,
        invalid_edges=invalid,
        b2b_nordea_verdict="READY WITH PLATFORM GAPS",
        r1_ready_for_audit=ready,
        remediation_children=tuple(remediation),
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
        "critical_authority_participants": report.critical_authority_participants,
        "non_authority_duplicate_participants": report.non_authority_duplicate_participants,
        "unknown_authority_participation": report.unknown_authority_participation,
        "non_critical_owner_scope_uncertainty": report.non_critical_owner_scope_uncertainty,
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
