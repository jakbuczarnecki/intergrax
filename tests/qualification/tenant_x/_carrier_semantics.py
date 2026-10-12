# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R2 class-field carrier family + provenance requirement (classification only)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._authority_semantics import (
    TenantAuthorityReviewMetrics,
    build_authority_review_metrics,
)
from tests.qualification.tenant_x._types import (
    TenantAuthorityRole,
    TenantAuthoritySemanticRole,
    TenantCarrierFamily,
    TenantCarrierProvenanceRequirement,
    TenantDiscoveryCandidate,
    TenantDomain,
    TenantIdentitySource,
    TenantPropagationRole,
    TenantProvenanceProofStatus,
    TenantSurfaceRecord,
)

_BOUNDARY_CRITICAL_FAMILIES: frozenset[TenantCarrierFamily] = frozenset(
    {
        TenantCarrierFamily.RUNTIME_EXECUTION,
        TenantCarrierFamily.GOVERNANCE,
        TenantCarrierFamily.STATE_DURABLE,
        TenantCarrierFamily.EVIDENCE_TRACE,
        TenantCarrierFamily.PROVIDER_CONFIG,
        TenantCarrierFamily.CREDENTIAL_REFERENCE,
        TenantCarrierFamily.ASYNC_CHILD,
    }
)

_FAMILY_UNKNOWN_PATH_PREFIX = "synthetic/carrier_family_unknown/"


def carrier_family_for_class_field(
    candidate: TenantDiscoveryCandidate,
    domain: TenantDomain,
) -> TenantCarrierFamily:
    path = candidate.module_path
    if path.startswith(_FAMILY_UNKNOWN_PATH_PREFIX):
        return TenantCarrierFamily.UNKNOWN
    if domain == TenantDomain.PLATFORM_GLOBAL:
        return TenantCarrierFamily.GLOBAL_NON_TENANT
    if path.startswith("applications/"):
        return TenantCarrierFamily.APPLICATION_BOUNDARY
    if domain == TenantDomain.GOVERNANCE:
        return TenantCarrierFamily.GOVERNANCE
    if domain in {
        TenantDomain.RUNTIME_REQUEST_EXECUTION,
        TenantDomain.TOOL_INVOCATION,
        TenantDomain.CONTROL_PLANE,
    }:
        return TenantCarrierFamily.RUNTIME_EXECUTION
    if domain in {
        TenantDomain.DURABLE_STATE,
        TenantDomain.RUNTIME_STATE,
        TenantDomain.CHECKPOINT,
        TenantDomain.RECOVERY,
        TenantDomain.RESUME,
        TenantDomain.REPLAY,
        TenantDomain.RETRY,
        TenantDomain.BACKUP_RESTORE,
    }:
        return TenantCarrierFamily.STATE_DURABLE
    if domain in {TenantDomain.EVIDENCE, TenantDomain.TRACE, TenantDomain.EVENTS}:
        return TenantCarrierFamily.EVIDENCE_TRACE
    if domain in {
        TenantDomain.INTEGRATION_RESOLUTION,
        TenantDomain.PROVIDER_INVOCATION,
        TenantDomain.PROVIDER_EXTENSIONS,
        TenantDomain.CONFIGURATION_PROFILES,
        TenantDomain.PLUGINS,
    }:
        return TenantCarrierFamily.PROVIDER_CONFIG
    if domain == TenantDomain.CREDENTIALS_SECRETS:
        return TenantCarrierFamily.CREDENTIAL_REFERENCE
    if domain in {
        TenantDomain.MEMORY,
        TenantDomain.CACHE,
        TenantDomain.RAG,
        TenantDomain.VECTOR_INDEX,
        TenantDomain.GRAPH_INDEX,
    }:
        return TenantCarrierFamily.MEMORY_INDEX
    if domain in {
        TenantDomain.QUEUES,
        TenantDomain.BACKGROUND_WORK,
        TenantDomain.CHILD_EXECUTION,
    } or path.startswith("agents/"):
        return TenantCarrierFamily.ASYNC_CHILD
    if "compat" in path or "legacy" in path or domain == TenantDomain.COMPATIBILITY_ASSESSMENT:
        return TenantCarrierFamily.COMPATIBILITY
    if domain in {
        TenantDomain.IDENTITY_AUTH,
        TenantDomain.OBSERVABILITY,
        TenantDomain.DIAGNOSTICS,
        TenantDomain.EXTERNAL_OPERATIONS,
        TenantDomain.CONTEXT_PROMPT,
    }:
        if path.startswith("applications/"):
            return TenantCarrierFamily.APPLICATION_BOUNDARY
        if "health" in path or "metrics" in path:
            return TenantCarrierFamily.GLOBAL_NON_TENANT
    return TenantCarrierFamily.UNKNOWN


def provenance_requirement_for_family(
    family: TenantCarrierFamily,
) -> TenantCarrierProvenanceRequirement:
    if family == TenantCarrierFamily.GLOBAL_NON_TENANT:
        return TenantCarrierProvenanceRequirement.GLOBAL_NON_TENANT
    if family in {TenantCarrierFamily.RUNTIME_EXECUTION, TenantCarrierFamily.GOVERNANCE}:
        return TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_BOUNDARY
    if family == TenantCarrierFamily.EVIDENCE_TRACE:
        return TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_USE_SITE
    if family in {
        TenantCarrierFamily.STATE_DURABLE,
        TenantCarrierFamily.APPLICATION_BOUNDARY,
        TenantCarrierFamily.MEMORY_INDEX,
        TenantCarrierFamily.COMPATIBILITY,
    }:
        return TenantCarrierProvenanceRequirement.PASSIVE_CARRIER_ONLY
    if family in {
        TenantCarrierFamily.PROVIDER_CONFIG,
        TenantCarrierFamily.CREDENTIAL_REFERENCE,
        TenantCarrierFamily.ASYNC_CHILD,
    }:
        return TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_USE_SITE
    return TenantCarrierProvenanceRequirement.UNKNOWN_REQUIREMENT


def is_boundary_critical_family(family: TenantCarrierFamily) -> bool:
    return family in _BOUNDARY_CRITICAL_FAMILIES


def unknown_carrier_family_cluster(module_path: str) -> str:
    if module_path.startswith("applications/"):
        return "application_boundary"
    if module_path.startswith("intergrax/contracts/"):
        return "contracts"
    if module_path.startswith("intergrax/context/"):
        return "context_prompt"
    if "auth" in module_path or "identity" in module_path:
        return "identity_auth"
    if module_path.startswith("intergrax/core/"):
        return "platform_core"
    return "unknown_cluster"


@dataclass(frozen=True, slots=True)
class BoundaryCriticalFamilyReport:
    family: str
    total_carriers: int
    provenance_required: int
    current_head_proven: int
    historical_evidence_only: int
    provenance_missing: int
    authority_carrier: int
    authority_validator: int
    authority_derivation: int
    authority_narrowing: int
    authority_mint_rewrite: int
    authority_unknown: int


def build_boundary_critical_family_reports(
    class_field_records: tuple[TenantSurfaceRecord, ...],
) -> tuple[BoundaryCriticalFamilyReport, ...]:
    by_family: dict[TenantCarrierFamily, list[TenantSurfaceRecord]] = {}
    for record in class_field_records:
        if not is_boundary_critical_family(record.carrier_family):
            continue
        by_family.setdefault(record.carrier_family, []).append(record)

    reports: list[BoundaryCriticalFamilyReport] = []
    for family in TenantCarrierFamily:
        if family not in by_family:
            continue
        records = by_family[family]
        req = sum(
            1
            for r in records
            if r.carrier_provenance_requirement
            in {
                TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_BOUNDARY,
                TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_USE_SITE,
            }
        )
        head = sum(1 for r in records if r.provenance_proof_status == TenantProvenanceProofStatus.PROVEN_CURRENT_HEAD)
        hist = sum(1 for r in records if r.provenance_proof_status == TenantProvenanceProofStatus.HISTORICAL_ONLY)
        missing = sum(1 for r in records if r.provenance_proof_status == TenantProvenanceProofStatus.MISSING)

        reports.append(
            BoundaryCriticalFamilyReport(
                family=family.value,
                total_carriers=len(records),
                provenance_required=req,
                current_head_proven=head,
                historical_evidence_only=hist,
                provenance_missing=missing,
                authority_carrier=sum(
                    1 for r in records if r.authority_semantic_role == TenantAuthoritySemanticRole.CARRIER
                ),
                authority_validator=sum(
                    1 for r in records if r.authority_semantic_role == TenantAuthoritySemanticRole.VALIDATOR
                ),
                authority_derivation=sum(
                    1 for r in records if r.authority_semantic_role == TenantAuthoritySemanticRole.DERIVER
                ),
                authority_narrowing=sum(
                    1
                    for r in records
                    if r.authority_semantic_role == TenantAuthoritySemanticRole.SCOPE_NARROWER
                ),
                authority_mint_rewrite=sum(
                    1
                    for r in records
                    if r.authority_semantic_role == TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY
                ),
                authority_unknown=sum(
                    1 for r in records if r.authority_semantic_role == TenantAuthoritySemanticRole.UNKNOWN
                ),
            )
        )
    return tuple(reports)


def identity_provenance_known(record: TenantSurfaceRecord) -> bool:
    return record.identity_source != TenantIdentitySource.UNKNOWN


@dataclass(frozen=True, slots=True)
class TenantCarrierFamilyBucket:
    family: TenantCarrierFamily
    total: int
    provenance_known: int
    provenance_unknown: int
    boundary_critical_unknown: int
    passive_carrier_unknown: int


@dataclass(frozen=True, slots=True)
class TenantCarrierSemanticsReport:
    class_field_carrier_total: int
    identity_carrier_count: int
    unknown_propagation_roles: int
    identity_provenance_known: int
    identity_provenance_unknown: int
    boundary_critical_provenance_unknown: int
    passive_carrier_provenance_unknown: int
    generic_class_field_authority_count: int
    family_buckets: tuple[TenantCarrierFamilyBucket, ...]
    unknown_family_clusters: tuple[tuple[str, int], ...]
    provenance_required_count: int
    provenance_current_head_proven: int
    provenance_historical_only: int
    provenance_missing_count: int
    authority_review: TenantAuthorityReviewMetrics
    boundary_critical_reports: tuple[BoundaryCriticalFamilyReport, ...]


def build_carrier_semantics_report(
    inventory: tuple[TenantSurfaceRecord, ...],
    *,
    class_field_surface_ids: frozenset[str],
) -> TenantCarrierSemanticsReport:
    class_fields = tuple(r for r in inventory if r.surface_id in class_field_surface_ids)
    family_totals: dict[TenantCarrierFamily, list[TenantSurfaceRecord]] = {}
    for record in class_fields:
        family_totals.setdefault(record.carrier_family, []).append(record)

    buckets: list[TenantCarrierFamilyBucket] = []
    for family in TenantCarrierFamily:
        records = family_totals.get(family, [])
        prov_known = sum(1 for r in records if identity_provenance_known(r))
        prov_unknown = len(records) - prov_known
        bc_unknown = sum(
            1
            for r in records
            if not identity_provenance_known(r) and is_boundary_critical_family(family)
        )
        passive_unknown = sum(
            1
            for r in records
            if not identity_provenance_known(r)
            and r.carrier_provenance_requirement == TenantCarrierProvenanceRequirement.PASSIVE_CARRIER_ONLY
        )
        buckets.append(
            TenantCarrierFamilyBucket(
                family=family,
                total=len(records),
                provenance_known=prov_known,
                provenance_unknown=prov_unknown,
                boundary_critical_unknown=bc_unknown,
                passive_carrier_unknown=passive_unknown,
            )
        )

    generic_authority = sum(
        1
        for r in class_fields
        if r.authority_role == TenantAuthorityRole.TENANT_SCOPE_AUTHORITY
    )

    identity_carriers = sum(
        1 for r in inventory if r.propagation_role == TenantPropagationRole.IDENTITY_CARRIER
    )
    unknown_prop = sum(1 for r in inventory if r.propagation_role == TenantPropagationRole.UNKNOWN)
    prov_known_all = sum(1 for r in class_fields if identity_provenance_known(r))

    bc_prov_unknown = sum(
        1
        for r in class_fields
        if not identity_provenance_known(r) and is_boundary_critical_family(r.carrier_family)
    )
    passive_prov_unknown = sum(
        1
        for r in class_fields
        if not identity_provenance_known(r)
        and r.carrier_provenance_requirement == TenantCarrierProvenanceRequirement.PASSIVE_CARRIER_ONLY
    )

    cluster_counts: dict[str, int] = {}
    for record in class_fields:
        if record.carrier_family != TenantCarrierFamily.UNKNOWN:
            continue
        cluster = unknown_carrier_family_cluster(record.module_path)
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

    prov_required = sum(
        1
        for r in class_fields
        if r.carrier_provenance_requirement
        in {
            TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_BOUNDARY,
            TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_USE_SITE,
        }
    )
    prov_head = sum(
        1 for r in class_fields if r.provenance_proof_status == TenantProvenanceProofStatus.PROVEN_CURRENT_HEAD
    )
    prov_hist = sum(
        1 for r in class_fields if r.provenance_proof_status == TenantProvenanceProofStatus.HISTORICAL_ONLY
    )
    prov_missing = sum(
        1 for r in class_fields if r.provenance_proof_status == TenantProvenanceProofStatus.MISSING
    )

    authority_review = build_authority_review_metrics(inventory)
    bc_reports = build_boundary_critical_family_reports(tuple(class_fields))

    return TenantCarrierSemanticsReport(
        class_field_carrier_total=len(class_fields),
        identity_carrier_count=identity_carriers,
        unknown_propagation_roles=unknown_prop,
        identity_provenance_known=prov_known_all,
        identity_provenance_unknown=len(class_fields) - prov_known_all,
        boundary_critical_provenance_unknown=bc_prov_unknown,
        passive_carrier_provenance_unknown=passive_prov_unknown,
        generic_class_field_authority_count=generic_authority,
        family_buckets=tuple(buckets),
        unknown_family_clusters=tuple(sorted(cluster_counts.items())),
        provenance_required_count=prov_required,
        provenance_current_head_proven=prov_head,
        provenance_historical_only=prov_hist,
        provenance_missing_count=prov_missing,
        authority_review=authority_review,
        boundary_critical_reports=bc_reports,
    )
