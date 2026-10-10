# © Artur Czarnecki. All rights reserved.

"""Map accepted P0 inventory surfaces to COMPAT-X-R1 version-policy classifications."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY
from tests.qualification.compat_x._compat_x_types import CompatSurfaceRecord, EvolutionState, ExposureFacet
from tests.qualification.compat_x._compat_x_versioning_policy import (
    COMPAT_X_CROSS_PLATFORM_POLICY_AUTHORITY,
    COMPAT_X_QUALIFICATION_ENFORCEMENT_ROOT,
    LaterStageDeferral,
    R1ComplianceState,
    VersionAuthorityDisposition,
    VersionAuthorityModel,
    VersionFieldRole,
    VersionIdentityCompliance,
    VersionIdentityScheme,
    VersionObligation,
    classify_version_field_role,
    infer_version_identity_scheme,
)

_DOMAIN_VERSION_OWNERS: dict[str, str] = {
    "registry.contract:": "intergrax/contracts/migrations/registry.py",
    "registry.runtime:": "intergrax/runtime/schema/registry.py",
    "registry.event:": "intergrax/runtime/events/payload_registry.py",
}

_COMPAT_SHIM_PREFIX = "intergrax/compat/"
_QUALIFICATION_PREFIX = "tests/qualification/"


@dataclass(frozen=True, slots=True)
class VersionPolicyClassification:
    surface_id: str
    version_obligation: VersionObligation
    version_identity_scheme: VersionIdentityScheme
    version_field_role: VersionFieldRole
    authority: VersionAuthorityModel
    version_identity_compliance: VersionIdentityCompliance
    later_stage_deferral: LaterStageDeferral
    r1_compliance: R1ComplianceState
    models_current_vs_supported_read_distinct: bool


def _parse_class_field(surface_id: str) -> tuple[str, str, str] | None:
    prefix = "semantic.class.field:"
    if not surface_id.startswith(prefix):
        return None
    body = surface_id[len(prefix) :]
    module_path, class_name, field_name = body.rsplit(":", 2)
    return module_path, class_name, field_name


_CANONICAL_ENVELOPE_OWNER_PATHS: frozenset[str] = frozenset(_DOMAIN_VERSION_OWNERS.values())


def resolve_domain_version_authority(
    record: CompatSurfaceRecord,
    obligation: VersionObligation,
) -> tuple[VersionAuthorityDisposition, str | None]:
    if ExposureFacet.COMPATIBILITY_ADAPTER in record.exposure_facets:
        owner_hint = (
            record.compatibility_policy_owner.split(" (")[0]
            if record.compatibility_policy_owner
            else ""
        )
        if (
            owner_hint
            and owner_hint != record.owner_module_path
            and not owner_hint.startswith(_COMPAT_SHIM_PREFIX)
            and owner_hint in _CANONICAL_ENVELOPE_OWNER_PATHS
        ):
            return VersionAuthorityDisposition.INHERITED_VERSION_OWNER, owner_hint
        return VersionAuthorityDisposition.NO_VERSION_AUTHORITY, None

    if obligation in {
        VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED,
        VersionObligation.NOT_APPLICABLE,
    }:
        return VersionAuthorityDisposition.NO_VERSION_AUTHORITY, None

    identity = record.semantic_identity
    for prefix, owner in _DOMAIN_VERSION_OWNERS.items():
        if identity.startswith(prefix):
            return VersionAuthorityDisposition.DOMAIN_VERSION_OWNER, owner

    if record.compatibility_policy_owner.startswith("intergrax/"):
        path = record.compatibility_policy_owner.split(" (")[0]
        if path.startswith(_COMPAT_SHIM_PREFIX):
            return VersionAuthorityDisposition.NO_VERSION_AUTHORITY, None
        return VersionAuthorityDisposition.DOMAIN_VERSION_OWNER, path

    owner = record.owner_module_path
    if owner.startswith(_COMPAT_SHIM_PREFIX) or owner.startswith(_QUALIFICATION_PREFIX):
        return VersionAuthorityDisposition.NO_VERSION_AUTHORITY, None
    return VersionAuthorityDisposition.DOMAIN_VERSION_OWNER, owner


def _scheme_for_record(
    record: CompatSurfaceRecord,
    field_role: VersionFieldRole,
    obligation: VersionObligation,
) -> VersionIdentityScheme:
    if obligation in {
        VersionObligation.NOT_APPLICABLE,
        VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED,
    }:
        return VersionIdentityScheme.NOT_APPLICABLE
    current = record.current_version or ""
    inferred = infer_version_identity_scheme(current)
    if obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED:
        return inferred
    if record.semantic_identity.startswith("registry."):
        return inferred
    if field_role == VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION:
        return inferred
    if field_role not in {VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION, VersionFieldRole.NOT_APPLICABLE}:
        return VersionIdentityScheme.NOT_APPLICABLE
    return VersionIdentityScheme.NOT_APPLICABLE


def compute_version_identity_compliance(
    record: CompatSurfaceRecord,
    obligation: VersionObligation,
    deferral: LaterStageDeferral,
    disposition: VersionAuthorityDisposition,
    r1_compliance: R1ComplianceState,
) -> VersionIdentityCompliance:
    if obligation in {
        VersionObligation.NOT_APPLICABLE,
        VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED,
    }:
        return VersionIdentityCompliance.NO_VERSION_REQUIRED
    if obligation == VersionObligation.VERSION_INHERITED_FROM_CANONICAL_ENVELOPE:
        cv = (record.current_version or "").strip()
        if disposition != VersionAuthorityDisposition.INHERITED_VERSION_OWNER or not cv or cv.lower() == "unknown":
            return VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
        return VersionIdentityCompliance.VERSION_INHERITED_AND_RESOLVED
    if r1_compliance == R1ComplianceState.BLOCKED_R2_VALIDATION:
        return VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
    if obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED:
        cv = (record.current_version or "").strip()
        missing = not cv or cv.lower() == "unknown"
        if missing:
            if deferral != LaterStageDeferral.NOT_APPLICABLE:
                return VersionIdentityCompliance.BLOCKED_LATER_STAGE
            return VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
        if disposition == VersionAuthorityDisposition.NO_VERSION_AUTHORITY:
            return VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
        return VersionIdentityCompliance.VERSION_PRESENT_AND_OWNED
    return VersionIdentityCompliance.NO_VERSION_REQUIRED


def classify_inventory_surface(record: CompatSurfaceRecord) -> VersionPolicyClassification:
    surface_id = record.surface_id
    field_role = VersionFieldRole.NOT_APPLICABLE
    parsed = _parse_class_field(surface_id)
    if parsed is not None:
        _, _, field_name = parsed
        field_role = classify_version_field_role(field_name)

    obligation = VersionObligation.UNCLASSIFIED
    deferral = LaterStageDeferral.NOT_APPLICABLE
    compliance = R1ComplianceState.COMPLIANT

    if EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION in record.evolution_states:
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        compliance = R1ComplianceState.BLOCKED_R2_VALIDATION
        deferral = LaterStageDeferral.R2_PERSISTED_MIGRATION
        field_role = VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION
    elif surface_id.startswith("semantic.registry.contract:") or surface_id.startswith(
        "semantic.registry.runtime:"
    ):
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        if ExposureFacet.PERSISTED_SCHEMA in record.exposure_facets:
            deferral = LaterStageDeferral.R2_PERSISTED_MIGRATION
    elif surface_id.startswith("semantic.registry.event:"):
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        deferral = LaterStageDeferral.R3_EVENT_EVOLUTION
    elif ExposureFacet.EVENT_SCHEMA in record.exposure_facets and surface_id.startswith("semantic."):
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        deferral = LaterStageDeferral.R3_EVENT_EVOLUTION
    elif ExposureFacet.PLUGIN_PROVIDER_CONTRACT in record.exposure_facets and (
        surface_id.startswith("semantic.mechanism") or "plugin" in record.compatibility_policy_owner
    ):
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        deferral = LaterStageDeferral.R4_PLUGIN_PROVIDER
    elif ExposureFacet.COMPATIBILITY_ADAPTER in record.exposure_facets:
        if ExposureFacet.PUBLIC_STABLE in record.exposure_facets:
            obligation = VersionObligation.VERSION_INHERITED_FROM_CANONICAL_ENVELOPE
        else:
            obligation = VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED
            field_role = VersionFieldRole.NOT_APPLICABLE
    elif surface_id.startswith("semantic.migration:"):
        obligation = VersionObligation.NOT_APPLICABLE
        deferral = LaterStageDeferral.R2_PERSISTED_MIGRATION
    elif field_role == VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION:
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        if ExposureFacet.PUBLIC_STABLE in record.exposure_facets:
            deferral = LaterStageDeferral.NOT_APPLICABLE
        elif ExposureFacet.PERSISTED_SCHEMA in record.exposure_facets:
            deferral = LaterStageDeferral.R2_PERSISTED_MIGRATION
    elif field_role in {
        VersionFieldRole.OPTIMISTIC_CONCURRENCY,
        VersionFieldRole.BUSINESS_ENTITY_REVISION,
        VersionFieldRole.NON_SCHEMA_DOMAIN_VERSION,
    }:
        if ExposureFacet.PLUGIN_PROVIDER_CONTRACT in record.exposure_facets:
            obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
            deferral = LaterStageDeferral.R4_PLUGIN_PROVIDER
        else:
            obligation = VersionObligation.NOT_APPLICABLE
    elif surface_id.startswith("semantic.const:"):
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
    elif surface_id.startswith("semantic.schema.literal:"):
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
    elif ExposureFacet.PUBLIC_STABLE in record.exposure_facets:
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
    elif ExposureFacet.PERSISTED_SCHEMA in record.exposure_facets:
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        deferral = LaterStageDeferral.R2_PERSISTED_MIGRATION
        compliance = R1ComplianceState.BLOCKED_LATER_STAGE_ONLY
    elif ExposureFacet.PLUGIN_PROVIDER_CONTRACT in record.exposure_facets:
        obligation = VersionObligation.EXPLICIT_VERSION_REQUIRED
        deferral = LaterStageDeferral.R4_PLUGIN_PROVIDER
    else:
        obligation = VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED

    scheme = _scheme_for_record(record, field_role, obligation)
    if obligation == VersionObligation.UNCLASSIFIED:
        scheme = VersionIdentityScheme.UNCLASSIFIED

    if obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED:
        if ExposureFacet.PLUGIN_PROVIDER_CONTRACT in record.exposure_facets:
            deferral = LaterStageDeferral.R4_PLUGIN_PROVIDER
        elif ExposureFacet.EVENT_SCHEMA in record.exposure_facets:
            deferral = LaterStageDeferral.R3_EVENT_EVOLUTION
        elif (
            ExposureFacet.PERSISTED_SCHEMA in record.exposure_facets
            and compliance != R1ComplianceState.BLOCKED_R2_VALIDATION
        ):
            deferral = LaterStageDeferral.R2_PERSISTED_MIGRATION

    disposition, owner_path = resolve_domain_version_authority(record, obligation)
    identity_compliance = compute_version_identity_compliance(
        record, obligation, deferral, disposition, compliance
    )

    if (
        identity_compliance == VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
        and compliance == R1ComplianceState.COMPLIANT
        and ExposureFacet.PUBLIC_STABLE in record.exposure_facets
    ):
        compliance = R1ComplianceState.BLOCKED_LATER_STAGE_ONLY

    authority = VersionAuthorityModel(
        cross_platform_policy_authority=COMPAT_X_CROSS_PLATFORM_POLICY_AUTHORITY,
        qualification_enforcement_root=COMPAT_X_QUALIFICATION_ENFORCEMENT_ROOT,
        domain_version_disposition=disposition,
        domain_version_owner_path=owner_path,
        current_version_evidence_source=record.version_source,
    )

    distinct_read = record.semantic_identity.startswith("registry.")

    return VersionPolicyClassification(
        surface_id=surface_id,
        version_obligation=obligation,
        version_identity_scheme=scheme,
        version_field_role=field_role,
        authority=authority,
        version_identity_compliance=identity_compliance,
        later_stage_deferral=deferral,
        r1_compliance=compliance,
        models_current_vs_supported_read_distinct=distinct_read,
    )


@lru_cache(maxsize=1)
def build_version_policy_classifications() -> tuple[VersionPolicyClassification, ...]:
    return tuple(classify_inventory_surface(row) for row in COMPAT_X_INVENTORY)


COMPAT_X_R1_CLASSIFICATIONS: tuple[VersionPolicyClassification, ...] = build_version_policy_classifications()


def r1_classification_by_surface_id() -> dict[str, VersionPolicyClassification]:
    return {row.surface_id: row for row in COMPAT_X_R1_CLASSIFICATIONS}
