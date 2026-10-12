# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R2 class-field carrier family + provenance requirement (classification only)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.qualification.tenant_x._types import (
    TenantAuthorityRole,
    TenantCarrierFamily,
    TenantCarrierProvenanceRequirement,
    TenantDiscoveryCandidate,
    TenantDomain,
    TenantIdentitySource,
    TenantPropagationRole,
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
    duplicate_mint_rewrite_candidates: tuple[str, ...]
    family_buckets: tuple[TenantCarrierFamilyBucket, ...]


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

    mint_roles = frozenset(
        {
            TenantPropagationRole.CANONICAL_IDENTITY_SOURCE,
            TenantPropagationRole.TENANT_VALIDATOR,
            TenantPropagationRole.IDENTITY_DERIVATION,
        }
    )
    mint_candidates = tuple(
        r.surface_id
        for r in inventory
        if r.authority_role == TenantAuthorityRole.TENANT_SCOPE_AUTHORITY
        and r.propagation_role in mint_roles
        and "canonical_anchor_profile" not in "".join(r.evidence_refs)
        and r.surface_id not in class_field_surface_ids
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

    return TenantCarrierSemanticsReport(
        class_field_carrier_total=len(class_fields),
        identity_carrier_count=identity_carriers,
        unknown_propagation_roles=unknown_prop,
        identity_provenance_known=prov_known_all,
        identity_provenance_unknown=len(class_fields) - prov_known_all,
        boundary_critical_provenance_unknown=bc_prov_unknown,
        passive_carrier_provenance_unknown=passive_prov_unknown,
        generic_class_field_authority_count=generic_authority,
        duplicate_mint_rewrite_candidates=mint_candidates,
        family_buckets=tuple(buckets),
    )
