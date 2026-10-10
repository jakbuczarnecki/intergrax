# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1 qualification gates over P0 inventory + policy classifiers."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.migrations.registry import CONTRACT_SCHEMA_REGISTRY
from intergrax.runtime.schema.registry import RUNTIME_SCHEMA_REGISTRY

from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY
from tests.qualification.compat_x._compat_x_registry_analysis import analyze_registry_overlap
from tests.qualification.compat_x._compat_x_types import EvolutionState, ExposureFacet
from tests.qualification.compat_x._compat_x_versioning_classification import (
    COMPAT_X_R1_CLASSIFICATIONS,
    VersionPolicyClassification,
    classify_inventory_surface,
)
from tests.qualification.compat_x._compat_x_versioning_policy import (
    VersionObligation,
    VersionIdentityScheme,
)

_REQUIRED_OBLIGATION_FACETS: frozenset[ExposureFacet] = frozenset(
    {
        ExposureFacet.PUBLIC_STABLE,
        ExposureFacet.PERSISTED_SCHEMA,
        ExposureFacet.EVENT_SCHEMA,
        ExposureFacet.PLUGIN_PROVIDER_CONTRACT,
    }
)


@dataclass(frozen=True, slots=True)
class R1QualificationReport:
    classified_surfaces: int
    unclassified_obligation_count: int
    public_stable_without_version_obligation: int
    persisted_without_required_obligation: int
    event_without_obligation: int
    plugin_without_obligation: int
    ambiguous_owner_count: int
    duplicate_authority_count: int
    registry_version_conflicts: int
    persisted_without_version_finding_count: int
    version_policy_missing_evolution_count: int
    obligations_by_class: dict[str, int]
    schemes_by_class: dict[str, int]


def _obligation_required_for_record(row) -> bool:
    facets = row.exposure_facets
    if any(f in facets for f in _REQUIRED_OBLIGATION_FACETS):
        return True
    if EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION in row.evolution_states:
        return True
    return False


def _obligation_satisfies_requirement(obligation: VersionObligation) -> bool:
    return obligation in {
        VersionObligation.EXPLICIT_VERSION_REQUIRED,
        VersionObligation.VERSION_INHERITED_FROM_CANONICAL_ENVELOPE,
    }


def build_r1_qualification_report() -> R1QualificationReport:
    classifications = COMPAT_X_R1_CLASSIFICATIONS
    inventory_by_id = {row.surface_id: row for row in COMPAT_X_INVENTORY}

    unclassified = sum(1 for c in classifications if c.version_obligation == VersionObligation.UNCLASSIFIED)
    public_block = 0
    persisted_block = 0
    event_block = 0
    plugin_block = 0

    obligations: dict[str, int] = {}
    schemes: dict[str, int] = {}

    for clf in classifications:
        obligations[clf.version_obligation.value] = obligations.get(clf.version_obligation.value, 0) + 1
        schemes[clf.version_identity_scheme.value] = schemes.get(clf.version_identity_scheme.value, 0) + 1
        row = inventory_by_id[clf.surface_id]
        if ExposureFacet.PUBLIC_STABLE in row.exposure_facets and not _obligation_satisfies_requirement(
            clf.version_obligation
        ):
            public_block += 1
        if ExposureFacet.PERSISTED_SCHEMA in row.exposure_facets and not _obligation_satisfies_requirement(
            clf.version_obligation
        ):
            if EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION not in row.evolution_states:
                persisted_block += 1
        if ExposureFacet.EVENT_SCHEMA in row.exposure_facets and not _obligation_satisfies_requirement(
            clf.version_obligation
        ):
            event_block += 1
        if ExposureFacet.PLUGIN_PROVIDER_CONTRACT in row.exposure_facets and clf.version_obligation in {
            VersionObligation.UNCLASSIFIED,
            VersionObligation.NOT_APPLICABLE,
            VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED,
        }:
            if not row.surface_id.startswith("semantic.shim:"):
                plugin_block += 1

    duplicate = 0
    semantic_to_owner: dict[str, set[str]] = {}
    for row in COMPAT_X_INVENTORY:
        if not row.semantic_identity.startswith("registry."):
            continue
        owner = classify_inventory_surface(row).authority.domain_version_owner_path
        semantic_to_owner.setdefault(row.semantic_identity, set()).add(owner)
    for owners in semantic_to_owner.values():
        if len(owners) > 1:
            duplicate += 1

    ambiguous = sum(
        1
        for c in classifications
        if c.authority.domain_version_owner_path.startswith("FRAGMENTED")
        or c.authority.domain_version_owner_path.startswith("UNOWNED")
    )

    overlap = analyze_registry_overlap()
    persisted_findings = sum(
        1 for row in COMPAT_X_INVENTORY if EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION in row.evolution_states
    )
    policy_missing = sum(
        1 for row in COMPAT_X_INVENTORY if EvolutionState.VERSIONED_POLICY_MISSING in row.evolution_states
    )

    return R1QualificationReport(
        classified_surfaces=len(classifications),
        unclassified_obligation_count=unclassified,
        public_stable_without_version_obligation=public_block,
        persisted_without_required_obligation=persisted_block,
        event_without_obligation=event_block,
        plugin_without_obligation=plugin_block,
        ambiguous_owner_count=ambiguous,
        duplicate_authority_count=duplicate,
        registry_version_conflicts=len(overlap.version_conflicts),
        persisted_without_version_finding_count=persisted_findings,
        version_policy_missing_evolution_count=policy_missing,
        obligations_by_class=obligations,
        schemes_by_class=schemes,
    )


def registry_declared_versions_match() -> tuple[str, ...]:
    """Surfaces where inventory current_version disagrees with domain registry."""
    mismatches: list[str] = []
    for row in COMPAT_X_INVENTORY:
        if row.semantic_identity.startswith("registry.contract:"):
            name = row.semantic_identity.split(":", 1)[1]
            for entry in CONTRACT_SCHEMA_REGISTRY:
                if entry.contract_name == name and entry.current_version != row.current_version:
                    mismatches.append(row.surface_id)
        if row.semantic_identity.startswith("registry.runtime:"):
            key = row.semantic_identity.split(":", 1)[1]
            reg = RUNTIME_SCHEMA_REGISTRY.get(key)
            if reg is not None and reg != row.current_version:
                mismatches.append(row.surface_id)
    return tuple(mismatches)


def synthetic_public_stable_without_obligation() -> VersionPolicyClassification:
    from tests.qualification.compat_x._compat_x_types import (
        CompatDomain,
        CompatSurfaceRecord,
        MigrationMechanismClass,
        ShimClass,
    )

    row = CompatSurfaceRecord(
        surface_id="synthetic.r1.public.unversioned",
        semantic_identity="synthetic.public.unversioned",
        domain=CompatDomain.CONTRACTS,
        owner_module_path="synthetic/r1_probe.py",
        contract_schema_identity="SyntheticPublic",
        exposure_facets=frozenset({ExposureFacet.PUBLIC_STABLE}),
        persistence_wire="wire_or_transient",
        version_source="synthetic",
        current_version="",
        compatibility_policy_owner="synthetic",
        migration_owner_path="UNOWNED",
        deprecation_owner_path="UNOWNED",
        plugin_provider_relevant=False,
        tenant_relevant=False,
        evolution_states=frozenset({EvolutionState.VERSIONED_POLICY_MISSING}),
        migration_class=MigrationMechanismClass.NOT_APPLICABLE,
        shim_class=ShimClass.NOT_APPLICABLE,
        evidence_paths=("synthetic",),
    )
    clf = classify_inventory_surface(row)
    return VersionPolicyClassification(
        surface_id=clf.surface_id,
        version_obligation=VersionObligation.UNCLASSIFIED,
        version_identity_scheme=VersionIdentityScheme.UNCLASSIFIED,
        version_field_role=clf.version_field_role,
        authority=clf.authority,
        later_stage_deferral=clf.later_stage_deferral,
        r1_compliance=clf.r1_compliance,
        models_current_vs_supported_read_distinct=clf.models_current_vs_supported_read_distinct,
    )
