# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R3 authority semantic roles (carrier ≠ validator ≠ derivation ≠ mint)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tests.qualification.tenant_x._evidence import TenantAstEvidenceKind, evidence_kinds
from tests.qualification.tenant_x._types import (
    AmbientSubstitutionClass,
    TenantAuthoritySemanticRole,
    TenantCarrierProvenanceRequirement,
    TenantDiscoveryCandidate,
    TenantIdentitySource,
    TenantPropagationRole,
    TenantProvenanceProofStatus,
    TenantSurfaceRecord,
)

_EVID_VALIDATOR_SAME = TenantAstEvidenceKind.VALIDATOR_RETURNS_SAME_INPUT.value
_EVID_DERIVE = TenantAstEvidenceKind.TENANT_KWARG_DERIVED_FROM_PARENT.value
_EVID_LITERAL_REWRITE = TenantAstEvidenceKind.TENANT_KWARG_LITERAL_REWRITE.value
_EVID_SILENT_SUB = TenantAstEvidenceKind.TENANT_SILENT_DEFAULT_OR_SUBSTITUTION.value
_EVID_MINT_POSITIVE = TenantAstEvidenceKind.MINT_OR_REWRITE_POSITIVE.value
_EVID_DEFAULT_PROTECTED = TenantAstEvidenceKind.DEFAULT_FLOWS_TO_PROTECTED_OP.value


def resolve_authority_semantic_role(
    record: TenantSurfaceRecord,
    candidate: TenantDiscoveryCandidate | None,
) -> TenantAuthoritySemanticRole:
    kinds = evidence_kinds(candidate.evidence) if candidate is not None else frozenset()
    discovery = candidate.discovery_kind if candidate is not None else ""

    if TenantAstEvidenceKind.TENANT_SCOPE_WIDENING_ASSIGN.value in kinds:
        return TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY
    if _EVID_MINT_POSITIVE in kinds or _EVID_LITERAL_REWRITE in kinds:
        return TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY
    if _EVID_SILENT_SUB in kinds or (
        discovery.startswith("ambient_default") and _EVID_DEFAULT_PROTECTED in kinds
    ):
        return TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY

    if record.propagation_role == TenantPropagationRole.TENANT_SCOPE_NARROWING:
        return TenantAuthoritySemanticRole.SCOPE_NARROWER

    if _EVID_VALIDATOR_SAME in kinds:
        return TenantAuthoritySemanticRole.VALIDATOR

    if _EVID_DERIVE in kinds or record.identity_source in {
        TenantIdentitySource.DERIVED_FROM_PARENT,
        TenantIdentitySource.DERIVED_FROM_REQUEST,
    }:
        if record.propagation_role == TenantPropagationRole.IDENTITY_DERIVATION:
            return TenantAuthoritySemanticRole.DERIVER
        if discovery == "function.tenant_kwarg.site":
            return TenantAuthoritySemanticRole.DERIVER
    if discovery == "validator.marker" or record.propagation_role == TenantPropagationRole.TENANT_VALIDATOR:
        if _EVID_VALIDATOR_SAME in kinds:
            return TenantAuthoritySemanticRole.VALIDATOR
        if "canonical_anchor_profile" in "".join(record.evidence_refs):
            if record.propagation_role == TenantPropagationRole.TENANT_VALIDATOR:
                return TenantAuthoritySemanticRole.VALIDATOR
        return TenantAuthoritySemanticRole.UNKNOWN

    if record.propagation_role in {
        TenantPropagationRole.IDENTITY_CARRIER,
        TenantPropagationRole.TENANT_SCOPED_LOOKUP,
        TenantPropagationRole.TENANT_SCOPED_WRITE,
        TenantPropagationRole.TENANT_SCOPED_CACHE,
        TenantPropagationRole.TENANT_SCOPED_EVENT,
        TenantPropagationRole.TENANT_SCOPED_EVIDENCE,
    }:
        return TenantAuthoritySemanticRole.CARRIER

    if record.propagation_role == TenantPropagationRole.CANONICAL_IDENTITY_SOURCE:
        return TenantAuthoritySemanticRole.UNKNOWN

    if discovery == "class.field.tenant":
        return TenantAuthoritySemanticRole.CARRIER

    return TenantAuthoritySemanticRole.UNKNOWN


def resolve_provenance_proof_status(record: TenantSurfaceRecord) -> TenantProvenanceProofStatus:
    req = record.carrier_provenance_requirement
    if req in {
        TenantCarrierProvenanceRequirement.GLOBAL_NON_TENANT,
        TenantCarrierProvenanceRequirement.PASSIVE_CARRIER_ONLY,
    }:
        return TenantProvenanceProofStatus.NOT_REQUIRED
    if req == TenantCarrierProvenanceRequirement.UNKNOWN_REQUIREMENT:
        return TenantProvenanceProofStatus.MISSING

    proven_identity = record.identity_source != TenantIdentitySource.UNKNOWN
    has_structural = any(
        marker in "".join(record.evidence_refs)
        for marker in (
            "canonical_anchor_profile",
            "class_field_identity_carrier_only",
            "validated_before_lookup",
            "tenant_in_key_or_predicate",
            "explicit_parameter_carrier_only",
            "provenance:current_head",
        )
    )
    if "canonical_anchor_profile" in "".join(record.evidence_refs):
        return TenantProvenanceProofStatus.HISTORICAL_ONLY
    if proven_identity or has_structural:
        if record.identity_source in {
            TenantIdentitySource.DERIVED_FROM_PARENT,
            TenantIdentitySource.DERIVED_FROM_REQUEST,
            TenantIdentitySource.VALIDATED_ENVELOPE,
            TenantIdentitySource.CANONICAL_CONTRACT,
        }:
            return TenantProvenanceProofStatus.PROVEN_CURRENT_HEAD
    if req in {
        TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_BOUNDARY,
        TenantCarrierProvenanceRequirement.PROVENANCE_REQUIRED_AT_USE_SITE,
    }:
        return TenantProvenanceProofStatus.MISSING
    return TenantProvenanceProofStatus.NOT_REQUIRED


def resolve_ambient_substitution_class(record: TenantSurfaceRecord) -> AmbientSubstitutionClass | None:
    joined = "".join(record.evidence_refs)
    if not joined.startswith("TENANT-X-P0:ambient:") and _EVID_SILENT_SUB not in joined:
        return None
    if "GLOBAL_NON_TENANT" in joined:
        return AmbientSubstitutionClass.LEGITIMATE_GLOBAL_SURFACE
    if "COMPATIBILITY" in joined:
        return AmbientSubstitutionClass.COMPATIBILITY_BEHAVIOR
    if "TENANT_REQUIRED_SILENT_DEFAULT" in joined or _EVID_SILENT_SUB in joined:
        return AmbientSubstitutionClass.ILLEGAL_SILENT_SUBSTITUTION
    return AmbientSubstitutionClass.UNKNOWN


def finalize_surface_record(
    record: TenantSurfaceRecord,
    candidate: TenantDiscoveryCandidate,
) -> TenantSurfaceRecord:
    role = resolve_authority_semantic_role(record, candidate)
    proof = resolve_provenance_proof_status(record)
    ambient = resolve_ambient_substitution_class(record)
    return replace(
        record,
        authority_semantic_role=role,
        provenance_proof_status=proof,
        ambient_substitution_class=ambient,
    )


def is_positive_mint_rewrite(role: TenantAuthoritySemanticRole, record: TenantSurfaceRecord) -> bool:
    if role != TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY:
        return False
    joined = "".join(record.evidence_refs)
    return (
        "mint:literal_rewrite_kwarg" in joined
        or "mint:silent_substitution" in joined
        or "mint:scope_widening" in joined
        or "ambient:TENANT_REQUIRED_SILENT_DEFAULT" in joined
        or "mint:positive" in joined
        or ("ambient_default" in joined and "TENANT_REQUIRED_SILENT_DEFAULT" in joined)
    )


@dataclass(frozen=True, slots=True)
class TenantAuthorityReviewMetrics:
    validator_candidates: tuple[str, ...]
    derivation_candidates: tuple[str, ...]
    scope_narrowing_candidates: tuple[str, ...]
    actual_mint_rewrite_candidates: tuple[str, ...]
    unknown_authority_candidates: tuple[str, ...]
    unowned_mint_rewrite_candidates: tuple[str, ...]
    duplicate_authority_candidates: tuple[str, ...]


def build_authority_review_metrics(
    inventory: tuple[TenantSurfaceRecord, ...],
) -> TenantAuthorityReviewMetrics:
    validators: list[str] = []
    derivations: list[str] = []
    narrowing: list[str] = []
    mint: list[str] = []
    unknown_auth: list[str] = []
    mint_surfaces: list[TenantSurfaceRecord] = []

    for record in inventory:
        role = record.authority_semantic_role
        if role == TenantAuthoritySemanticRole.VALIDATOR:
            validators.append(record.surface_id)
        elif role == TenantAuthoritySemanticRole.DERIVER:
            derivations.append(record.surface_id)
        elif role == TenantAuthoritySemanticRole.SCOPE_NARROWER:
            narrowing.append(record.surface_id)
        elif role == TenantAuthoritySemanticRole.MINT_OR_REWRITE_AUTHORITY:
            if is_positive_mint_rewrite(role, record):
                mint.append(record.surface_id)
                mint_surfaces.append(record)
            else:
                unknown_auth.append(record.surface_id)
        elif role == TenantAuthoritySemanticRole.UNKNOWN:
            if record.propagation_role in {
                TenantPropagationRole.TENANT_VALIDATOR,
                TenantPropagationRole.CANONICAL_IDENTITY_SOURCE,
            }:
                unknown_auth.append(record.surface_id)
            elif record.authority_role.value == "TENANT_SCOPE_AUTHORITY":
                unknown_auth.append(record.surface_id)

    unowned: list[str] = []
    owners: dict[str, list[str]] = {}
    for record in mint_surfaces:
        owner = record.semantic_owner
        owners.setdefault(owner, []).append(record.surface_id)
        if owner in {"", "PLATFORM_CORE"} and "canonical_anchor_profile" not in "".join(record.evidence_refs):
            unowned.append(record.surface_id)

    duplicate: list[str] = []
    for owner, surfaces in owners.items():
        if len(surfaces) > 1 and owner not in {"PLATFORM_CORE", "OBSERVABILITY"}:
            duplicate.extend(surfaces)

    return TenantAuthorityReviewMetrics(
        validator_candidates=tuple(sorted(validators)),
        derivation_candidates=tuple(sorted(derivations)),
        scope_narrowing_candidates=tuple(sorted(narrowing)),
        actual_mint_rewrite_candidates=tuple(sorted(mint)),
        unknown_authority_candidates=tuple(sorted(set(unknown_auth))),
        unowned_mint_rewrite_candidates=tuple(sorted(set(unowned))),
        duplicate_authority_candidates=tuple(sorted(set(duplicate))),
    )

