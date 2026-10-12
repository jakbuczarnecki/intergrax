# © Artur Czarnecki. All rights reserved.

"""TENANT-X-P0-R3 authority semantic roles (carrier ≠ validator ≠ derivation ≠ mint)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from tests.qualification.tenant_x._evidence import TenantAstEvidenceKind, evidence_kinds
from tests.qualification.tenant_x._types import (
    AmbientSubstitutionClass,
    TenantAuthorityDecisionKind,
    TenantAuthorityOwnerDisposition,
    TenantAuthoritySemanticRole,
    TenantCarrierProvenanceRequirement,
    TenantDiscoveryCandidate,
    TenantDomain,
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


def classify_authority_decision_kind(record: TenantSurfaceRecord) -> TenantAuthorityDecisionKind:
    joined = "".join(record.evidence_refs)
    if "authority:decision:" in joined:
        for ref in record.evidence_refs:
            if ref.startswith("TENANT-X-P0:authority:decision:"):
                raw = ref.removeprefix("TENANT-X-P0:authority:decision:")
                try:
                    return TenantAuthorityDecisionKind(raw)
                except ValueError:
                    return TenantAuthorityDecisionKind.UNKNOWN
    if record.domain in {TenantDomain.RECOVERY, TenantDomain.RESUME} or (
        record.propagation_role == TenantPropagationRole.TENANT_SCOPED_RECOVERY
    ):
        return TenantAuthorityDecisionKind.RECOVERY_TENANT_RESTORE
    if record.propagation_role == TenantPropagationRole.TENANT_SCOPED_CHILD_OPERATION:
        return TenantAuthorityDecisionKind.CHILD_TENANT_DERIVATION
    if "mint:scope_widening" in joined:
        return TenantAuthorityDecisionKind.TENANT_SCOPE_REWRITE
    if (
        "ambient:TENANT_REQUIRED_SILENT_DEFAULT" in joined
        or "mint:silent_substitution" in joined
        or record.fail_closed_behavior.value == "SILENT_DEFAULT_SUBSTITUTION"
    ):
        return TenantAuthorityDecisionKind.DEFAULT_SUBSTITUTION
    if record.propagation_role == TenantPropagationRole.TENANT_SCOPED_PROVIDER_SELECTION:
        return TenantAuthorityDecisionKind.PROVIDER_TENANT_BINDING
    if record.domain in {
        TenantDomain.RUNTIME_REQUEST_EXECUTION,
        TenantDomain.IDENTITY_AUTH,
        TenantDomain.GOVERNANCE,
        TenantDomain.CONTROL_PLANE,
    } and ("mint:literal_rewrite_kwarg" in joined or "mint:positive" in joined):
        return TenantAuthorityDecisionKind.ROOT_TENANT_SELECTION
    if "mint:literal_rewrite_kwarg" in joined or "mint:positive" in joined:
        return TenantAuthorityDecisionKind.TENANT_SCOPE_REWRITE
    if "mint:" in joined:
        return TenantAuthorityDecisionKind.OTHER
    return TenantAuthorityDecisionKind.UNKNOWN


def resolve_authority_scope(record: TenantSurfaceRecord) -> str:
    for ref in record.evidence_refs:
        if ref.startswith("TENANT-X-P0:authority_scope:"):
            return ref.removeprefix("TENANT-X-P0:authority_scope:")
    if record.domain == TenantDomain.RECOVERY:
        return f"recovery:{record.domain.value}"
    return f"domain:{record.domain.value}"


def _mechanism_contract_key(record: TenantSurfaceRecord) -> str:
    for ref in record.evidence_refs:
        if ref.startswith("TENANT-X-P0:authority_mechanism:"):
            return ref.removeprefix("TENANT-X-P0:authority_mechanism:")
    joined = "".join(record.evidence_refs)
    if "canonical_anchor_profile" in joined:
        return f"canonical:{record.module_path}:{record.symbol}"
    if "mint:scope_widening" in joined:
        return f"{record.module_path}::scope_widening"
    if "ambient:TENANT_REQUIRED_SILENT_DEFAULT" in joined or "mint:silent_substitution" in joined:
        return f"{record.module_path}::default_substitution"
    if "mint:literal_rewrite_kwarg" in joined:
        return f"{record.module_path}::literal_rewrite::{record.symbol}"
    if "mint:positive" in joined:
        return f"{record.module_path}::mint_positive::{record.symbol}"
    return f"{record.module_path}::{record.symbol}"


def resolve_owner_disposition(record: TenantSurfaceRecord) -> TenantAuthorityOwnerDisposition:
    joined = "".join(record.evidence_refs)
    if "authority:unowned" in joined or "TENANT-X-P0:UNOWNED_AUTHORITY" in joined:
        return TenantAuthorityOwnerDisposition.UNOWNED_AUTHORITY
    if "owner:conflict" in joined or "TENANT-X-P0:owner:conflict" in joined:
        return TenantAuthorityOwnerDisposition.OWNER_CONFLICT
    if "owner:confirmed:" in joined or "TENANT-X-P0:owner:confirmed:" in joined:
        return TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
    if "canonical_anchor_profile" in joined:
        return TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
    owner = record.semantic_owner
    if owner in {"", "PLATFORM_CORE"}:
        return TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED
    if owner == "OBSERVABILITY":
        return TenantAuthorityOwnerDisposition.OWNER_CANDIDATE
    return TenantAuthorityOwnerDisposition.OWNER_CANDIDATE


def _merge_owner_dispositions(
    dispositions: frozenset[TenantAuthorityOwnerDisposition],
) -> TenantAuthorityOwnerDisposition:
    if TenantAuthorityOwnerDisposition.OWNER_CONFLICT in dispositions:
        return TenantAuthorityOwnerDisposition.OWNER_CONFLICT
    if TenantAuthorityOwnerDisposition.UNOWNED_AUTHORITY in dispositions:
        return TenantAuthorityOwnerDisposition.UNOWNED_AUTHORITY
    if len(dispositions) == 1:
        return next(iter(dispositions))
    if TenantAuthorityOwnerDisposition.OWNER_CONFIRMED in dispositions:
        if dispositions <= {
            TenantAuthorityOwnerDisposition.OWNER_CONFIRMED,
            TenantAuthorityOwnerDisposition.OWNER_CANDIDATE,
        }:
            return TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
        return TenantAuthorityOwnerDisposition.OWNER_CONFLICT
    if dispositions == {
        TenantAuthorityOwnerDisposition.OWNER_CANDIDATE,
        TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED,
    }:
        return TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED
    return TenantAuthorityOwnerDisposition.OWNER_CONFLICT


@dataclass(frozen=True, slots=True)
class TenantAuthorityMechanism:
    mechanism_id: str
    semantic_owner: str
    authority_scope: str
    decision_kind: TenantAuthorityDecisionKind
    owner_disposition: TenantAuthorityOwnerDisposition
    surfaces: tuple[str, ...]


def _mechanism_group_key(record: TenantSurfaceRecord) -> tuple[str, TenantAuthorityDecisionKind, str, str]:
    decision = classify_authority_decision_kind(record)
    scope = resolve_authority_scope(record)
    contract = _mechanism_contract_key(record)
    owner = record.semantic_owner or "PLATFORM_CORE"
    return (owner, decision, scope, contract)


def build_authority_mechanisms(
    mint_surfaces: tuple[TenantSurfaceRecord, ...],
) -> tuple[TenantAuthorityMechanism, ...]:
    grouped: dict[tuple[str, TenantAuthorityDecisionKind, str, str], list[TenantSurfaceRecord]] = {}
    for record in mint_surfaces:
        grouped.setdefault(_mechanism_group_key(record), []).append(record)

    mechanisms: list[TenantAuthorityMechanism] = []
    for (owner, decision, scope, contract), records in sorted(grouped.items()):
        surface_ids = tuple(sorted({r.surface_id for r in records}))
        dispositions = frozenset(resolve_owner_disposition(r) for r in records)
        disposition = _merge_owner_dispositions(dispositions)
        mechanism_id = f"{owner}|{decision.value}|{scope}|{contract}"
        mechanisms.append(
            TenantAuthorityMechanism(
                mechanism_id=mechanism_id,
                semantic_owner=owner,
                authority_scope=scope,
                decision_kind=decision,
                owner_disposition=disposition,
                surfaces=surface_ids,
            )
        )
    return tuple(mechanisms)


def duplicate_authority_mechanism_ids(
    mechanisms: tuple[TenantAuthorityMechanism, ...],
) -> tuple[str, ...]:
    by_decision_scope: dict[tuple[TenantAuthorityDecisionKind, str], list[TenantAuthorityMechanism]] = {}
    for mechanism in mechanisms:
        key = (mechanism.decision_kind, mechanism.authority_scope)
        by_decision_scope.setdefault(key, []).append(mechanism)

    duplicate_ids: list[str] = []
    for group in by_decision_scope.values():
        if len(group) < 2:
            continue
        owners = {m.semantic_owner for m in group}
        if len(owners) < 2:
            continue
        competing = [
            m
            for m in group
            if m.owner_disposition
            not in {
                TenantAuthorityOwnerDisposition.UNOWNED_AUTHORITY,
            }
        ]
        if len({m.semantic_owner for m in competing}) < 2:
            continue
        duplicate_ids.extend(m.mechanism_id for m in competing)
    return tuple(sorted(set(duplicate_ids)))


@dataclass(frozen=True, slots=True)
class TenantAuthorityReviewMetrics:
    validator_candidates: tuple[str, ...]
    derivation_candidates: tuple[str, ...]
    scope_narrowing_candidates: tuple[str, ...]
    actual_mint_rewrite_surfaces: tuple[str, ...]
    unknown_authority_candidates: tuple[str, ...]
    authority_mechanisms: tuple[TenantAuthorityMechanism, ...]
    owner_confirmed: int
    owner_candidate: int
    owner_unresolved: int
    owner_conflict: int
    unowned_authority_mechanisms: tuple[str, ...]
    duplicate_authority_mechanisms: tuple[str, ...]

    @property
    def actual_mint_rewrite_candidates(self) -> tuple[str, ...]:
        """Compatibility alias — identical to ``actual_mint_rewrite_surfaces``."""
        return self.actual_mint_rewrite_surfaces

    @property
    def unowned_mint_rewrite_candidates(self) -> tuple[str, ...]:
        """Deprecated: surfaces on UNOWNED mechanisms (not unresolved-owner surfaces)."""
        unowned = set(self.unowned_authority_mechanisms)
        return tuple(
            sorted(
                sid
                for m in self.authority_mechanisms
                if m.mechanism_id in unowned
                for sid in m.surfaces
            )
        )

    @property
    def duplicate_authority_candidates(self) -> tuple[str, ...]:
        """Deprecated: surfaces belonging to competing duplicate mechanisms."""
        dup = set(self.duplicate_authority_mechanisms)
        return tuple(
            sorted(
                sid
                for m in self.authority_mechanisms
                if m.mechanism_id in dup
                for sid in m.surfaces
            )
        )


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

    mechanisms = build_authority_mechanisms(tuple(mint_surfaces))
    dup_ids = duplicate_authority_mechanism_ids(mechanisms)
    unowned_ids = tuple(
        sorted(m.mechanism_id for m in mechanisms if m.owner_disposition == TenantAuthorityOwnerDisposition.UNOWNED_AUTHORITY)
    )

    owner_confirmed = sum(
        1 for m in mechanisms if m.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFIRMED
    )
    owner_candidate = sum(
        1 for m in mechanisms if m.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CANDIDATE
    )
    owner_unresolved = sum(
        1 for m in mechanisms if m.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_UNRESOLVED
    )
    owner_conflict = sum(
        1 for m in mechanisms if m.owner_disposition == TenantAuthorityOwnerDisposition.OWNER_CONFLICT
    )

    return TenantAuthorityReviewMetrics(
        validator_candidates=tuple(sorted(validators)),
        derivation_candidates=tuple(sorted(derivations)),
        scope_narrowing_candidates=tuple(sorted(narrowing)),
        actual_mint_rewrite_surfaces=tuple(sorted(mint)),
        unknown_authority_candidates=tuple(sorted(set(unknown_auth))),
        authority_mechanisms=mechanisms,
        owner_confirmed=owner_confirmed,
        owner_candidate=owner_candidate,
        owner_unresolved=owner_unresolved,
        owner_conflict=owner_conflict,
        unowned_authority_mechanisms=unowned_ids,
        duplicate_authority_mechanisms=dup_ids,
    )


def summarize_authority_mechanism_review(
    metrics: TenantAuthorityReviewMetrics,
) -> dict[str, int | tuple[str, ...]]:
    by_decision: dict[str, int] = {}
    for mechanism in metrics.authority_mechanisms:
        by_decision[mechanism.decision_kind.value] = by_decision.get(mechanism.decision_kind.value, 0) + 1
    collapsed = sum(len(m.surfaces) - 1 for m in metrics.authority_mechanisms if len(m.surfaces) > 1)
    return {
        "positive_mint_rewrite_surfaces": len(metrics.actual_mint_rewrite_surfaces),
        "authority_mechanisms_total": len(metrics.authority_mechanisms),
        "mechanisms_by_decision_kind": tuple(f"{k}={v}" for k, v in sorted(by_decision.items())),
        "owner_confirmed": metrics.owner_confirmed,
        "owner_candidate": metrics.owner_candidate,
        "owner_unresolved": metrics.owner_unresolved,
        "owner_conflict": metrics.owner_conflict,
        "duplicate_authority_mechanisms": len(metrics.duplicate_authority_mechanisms),
        "surfaces_collapsed_into_shared_mechanism": collapsed,
    }
