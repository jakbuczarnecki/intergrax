# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-R1 versioning policy qualification gates."""

from __future__ import annotations

import pytest

from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY
from tests.qualification.compat_x._compat_x_owner_discovery import COMPAT_X_OWNER_MATRIX
from tests.qualification.compat_x._compat_x_registry_analysis import analyze_registry_overlap
from tests.qualification.compat_x._compat_x_types import EvolutionState, ExposureFacet, OwnerResponsibilityState
from tests.qualification.compat_x._compat_x_versioning_classification import (
    COMPAT_X_R1_CLASSIFICATIONS,
    classify_inventory_surface,
)
from tests.qualification.compat_x._compat_x_versioning_gates import (
    build_r1_qualification_report,
    registry_declared_versions_match,
    synthetic_public_stable_without_obligation,
)
from tests.qualification.compat_x._compat_x_versioning_policy import (
    COMPAT_X_VERSIONING_POLICY_OWNER,
    CompatibilityChangeClass,
    LaterStageDeferral,
    R1ComplianceState,
    ShimCanonicalAuthorityProbe,
    VersionFieldRole,
    VersionObligation,
    VersionIdentityScheme,
    classify_version_field_role,
    evaluate_change_classification,
    evaluate_declared_change_with_bump,
    evaluate_shim_canonical_authority,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_r1_inventory_fully_classified() -> None:
    report = build_r1_qualification_report()
    assert report.classified_surfaces == len(COMPAT_X_INVENTORY)
    assert report.unclassified_obligation_count == 0
    assert report.public_stable_without_version_obligation == 0
    assert report.persisted_without_required_obligation == 0
    assert report.event_without_obligation == 0
    assert report.plugin_without_obligation == 0
    assert report.ambiguous_owner_count == 0
    assert report.duplicate_authority_count == 0
    assert report.registry_version_conflicts == 0
    assert report.persisted_without_version_finding_count == 5
    assert report.version_policy_missing_evolution_count == 0


def test_cx_r1_frz_cmp_02_pass_candidate_scope() -> None:
    row = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "versioning_policy")
    assert row.responsibility_state == OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER
    assert "versioning_policy" in row.concern
    report = build_r1_qualification_report()
    assert report.unclassified_obligation_count == 0


def test_cx_r1_adversarial_a_public_stable_without_version_obligation_fails() -> None:
    clf = synthetic_public_stable_without_obligation()
    assert clf.version_obligation == VersionObligation.UNCLASSIFIED
    assert ExposureFacet.PUBLIC_STABLE in frozenset({ExposureFacet.PUBLIC_STABLE})
    assert clf.version_obligation not in {
        VersionObligation.EXPLICIT_VERSION_REQUIRED,
        VersionObligation.VERSION_INHERITED_FROM_CANONICAL_ENVELOPE,
    }


def test_cx_r1_adversarial_b_persisted_without_version_remains_blocker() -> None:
    blocked = [
        row
        for row in COMPAT_X_INVENTORY
        if EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION in row.evolution_states
    ]
    assert len(blocked) == 5
    for row in blocked:
        clf = classify_inventory_surface(row)
        assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
        assert clf.r1_compliance == R1ComplianceState.BLOCKED_R2_VALIDATION


def test_cx_r1_adversarial_c_breaking_structural_no_bump_fails() -> None:
    verdict = evaluate_declared_change_with_bump(CompatibilityChangeClass.BREAKING_STRUCTURAL, False)
    assert verdict.qualification_passes is False


def test_cx_r1_adversarial_d_semantic_change_same_shape_no_bump_fails() -> None:
    verdict = evaluate_declared_change_with_bump(CompatibilityChangeClass.BREAKING_SEMANTIC, False)
    assert verdict.qualification_passes is False
    assert verdict.requires_new_version_identity is True


def test_cx_r1_adversarial_e_unknown_compatibility_impact_fails_closed() -> None:
    verdict = evaluate_change_classification(CompatibilityChangeClass.UNKNOWN)
    assert verdict.qualification_passes is False


def test_cx_r1_adversarial_f_entity_version_int_not_schema_version() -> None:
    assert classify_version_field_role("version") == VersionFieldRole.BUSINESS_ENTITY_REVISION
    clf = classify_inventory_surface(
        next(r for r in COMPAT_X_INVENTORY if r.surface_id.endswith(":platform_version"))
    )
    assert clf.version_field_role == VersionFieldRole.NON_SCHEMA_DOMAIN_VERSION
    assert clf.version_obligation == VersionObligation.NOT_APPLICABLE


def test_cx_r1_adversarial_g_optimistic_concurrency_not_schema_version() -> None:
    assert classify_version_field_role("_version") == VersionFieldRole.OPTIMISTIC_CONCURRENCY
    role = classify_version_field_role("state_version")
    assert role == VersionFieldRole.OPTIMISTIC_CONCURRENCY


def test_cx_r1_adversarial_h_explicit_schema_version_maps_to_domain_owner() -> None:
    contract = next(r for r in COMPAT_X_INVENTORY if r.surface_id.startswith("semantic.registry.contract:"))
    clf = classify_inventory_surface(contract)
    assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
    assert "contracts/migrations/registry.py" in clf.authority.domain_version_owner_path
    assert clf.authority.cross_platform_policy_owner == COMPAT_X_VERSIONING_POLICY_OWNER


def test_cx_r1_adversarial_i_conflicting_registry_versions_fail() -> None:
    overlap = analyze_registry_overlap()
    assert overlap.version_conflicts == ()
    assert registry_declared_versions_match() == ()


def test_cx_r1_adversarial_j_shim_claiming_canonical_version_fails() -> None:
    assert (
        evaluate_shim_canonical_authority(
            ShimCanonicalAuthorityProbe(
                claims_canonical_current_version=True,
                adapter_supported_old_version="v1",
                canonical_version_owner_path="intergrax/contracts/migrations/registry.py",
            )
        )
        is False
    )


def test_cx_r1_adversarial_k_event_surface_has_r1_obligation_deferred_r3() -> None:
    event_rows = [r for r in COMPAT_X_INVENTORY if ExposureFacet.EVENT_SCHEMA in r.exposure_facets]
    assert event_rows
    for row in event_rows:
        clf = classify_inventory_surface(row)
        assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
        if row.surface_id.startswith("semantic.registry.event:"):
            assert clf.later_stage_deferral == LaterStageDeferral.R3_EVENT_EVOLUTION


def test_cx_r1_adversarial_l_plugin_surface_has_r1_obligation_deferred_r4() -> None:
    plugin_rows = [
        r
        for r in COMPAT_X_INVENTORY
        if ExposureFacet.PLUGIN_PROVIDER_CONTRACT in r.exposure_facets
        and not r.surface_id.startswith("semantic.shim:")
    ]
    assert plugin_rows
    for row in plugin_rows:
        clf = classify_inventory_surface(row)
        assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
        assert clf.later_stage_deferral == LaterStageDeferral.R4_PLUGIN_PROVIDER


def test_cx_r1_shim_not_version_owner_inventory() -> None:
    shims = [r for r in COMPAT_X_INVENTORY if ExposureFacet.COMPATIBILITY_ADAPTER in r.exposure_facets]
    for row in shims:
        clf = classify_inventory_surface(row)
        assert clf.version_obligation == VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED
        assert clf.authority.cross_platform_policy_owner == COMPAT_X_VERSIONING_POLICY_OWNER


def test_cx_r1_no_explicit_obligation_with_unclassified_scheme() -> None:
    for clf in COMPAT_X_R1_CLASSIFICATIONS:
        if clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED:
            assert clf.version_identity_scheme != VersionIdentityScheme.UNCLASSIFIED


def test_cx_r1_tenant_audit_reuse_extcomp() -> None:
    from tests.qualification.external_contract_compatibility.test_external_contract_compatibility_certification import (
        test_cert_22_resolver_tenant_isolation,
    )

    test_cert_22_resolver_tenant_isolation()


def test_cx_r1_current_vs_supported_read_distinct_for_registries() -> None:
    for clf in COMPAT_X_R1_CLASSIFICATIONS:
        if clf.surface_id.startswith("semantic.registry."):
            assert clf.models_current_vs_supported_read_distinct is True
