# © Artur Czarnecki. All rights reserved.

"""Map accepted P0 inventory surfaces to COMPAT-X-R1 version-policy classifications."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY
from tests.qualification.compat_x._compat_x_types import CompatSurfaceRecord, EvolutionState, ExposureFacet
from tests.qualification.compat_x._compat_x_versioning_policy import (
    COMPAT_X_VERSIONING_POLICY_OWNER,
    LaterStageDeferral,
    R1ComplianceState,
    VersionAuthorityModel,
    VersionFieldRole,
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


@dataclass(frozen=True, slots=True)
class VersionPolicyClassification:
    surface_id: str
    version_obligation: VersionObligation
    version_identity_scheme: VersionIdentityScheme
    version_field_role: VersionFieldRole
    authority: VersionAuthorityModel
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


def _domain_owner_for_surface(record: CompatSurfaceRecord) -> str:
    identity = record.semantic_identity
    for prefix, owner in _DOMAIN_VERSION_OWNERS.items():
        if identity.startswith(prefix):
            return owner
    if record.compatibility_policy_owner.startswith("intergrax/"):
        return record.compatibility_policy_owner.split(" (")[0]
    return record.owner_module_path


def _scheme_for_record(
    record: CompatSurfaceRecord,
    field_role: VersionFieldRole,
    obligation: VersionObligation,
) -> VersionIdentityScheme:
    if obligation == VersionObligation.NOT_APPLICABLE:
        return VersionIdentityScheme.NOT_APPLICABLE
    if record.semantic_identity.startswith("registry."):
        return infer_version_identity_scheme(record.current_version)
    if field_role == VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION:
        inferred = infer_version_identity_scheme(record.current_version or "")
        if inferred == VersionIdentityScheme.UNCLASSIFIED:
            return VersionIdentityScheme.SCHEMA_ID_GENERATION
        return inferred
    if obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED and field_role in {
        VersionFieldRole.NON_SCHEMA_DOMAIN_VERSION,
        VersionFieldRole.NOT_APPLICABLE,
    }:
        inferred = infer_version_identity_scheme(record.current_version or "")
        if inferred != VersionIdentityScheme.UNCLASSIFIED:
            return inferred
        return VersionIdentityScheme.EXTERNALLY_DEFINED_VERSION
    if field_role not in {VersionFieldRole.SCHEMA_CONTRACT_EVOLUTION, VersionFieldRole.NOT_APPLICABLE}:
        return VersionIdentityScheme.NOT_APPLICABLE
    return VersionIdentityScheme.NOT_APPLICABLE


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

    domain_owner = _domain_owner_for_surface(record)
    authority = VersionAuthorityModel(
        cross_platform_policy_owner=COMPAT_X_VERSIONING_POLICY_OWNER,
        domain_version_owner_path=domain_owner,
        current_version_source=record.version_source,
    )

    distinct_read = record.semantic_identity.startswith("registry.")

    return VersionPolicyClassification(
        surface_id=surface_id,
        version_obligation=obligation,
        version_identity_scheme=scheme,
        version_field_role=field_role,
        authority=authority,
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
