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
    classification_r1_qualification_passes,
    frz_cmp_02_pass_candidate,
    inventory_shim_canonical_authority_violations,
    public_stable_noncompliant_version_surfaces,
    registry_declared_versions_match,
    synthetic_public_stable_explicit_version,
    synthetic_public_stable_missing_version,
)
from tests.qualification.compat_x._compat_x_versioning_classification import compute_version_identity_compliance
from tests.qualification.compat_x._compat_x_versioning_policy import (
    COMPAT_X_CROSS_PLATFORM_POLICY_AUTHORITY,
    COMPAT_X_QUALIFICATION_ENFORCEMENT_ROOT,
    CompatibilityChangeClass,
    FamilyCompatibilityPolicy,
    LaterStageDeferral,
    R1ComplianceState,
    ShimCanonicalAuthorityProbe,
    VersionAuthorityDisposition,
    VersionFieldRole,
    VersionIdentityCompliance,
    VersionObligation,
    VersionIdentityScheme,
    classify_version_field_role,
    cross_platform_policy_is_qualification_module,
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
    assert report.public_stable_missing_version_identity == 0
    assert report.public_stable_noncompliant_version_identity_count == 0
    assert report.public_stable_noncompliant_surface_ids == ()
    assert report.static_version_reference_conflict_count == 0
    assert report.persisted_without_required_obligation == 0
    assert report.event_without_obligation == 0
    assert report.plugin_without_obligation == 0
    assert report.ambiguous_owner_count == 0
    assert report.duplicate_authority_count == 0
    assert report.registry_version_conflicts == 0
    assert report.persisted_without_version_finding_count == 5
    assert report.version_policy_missing_evolution_count == 0
    assert report.shim_canonical_authority_count == 0
    assert report.qualification_as_semantic_policy_owner == 0


def test_cx_r1_frz_cmp_02_pass_candidate_scope() -> None:
    row = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "versioning_policy")
    assert row.responsibility_state == OwnerResponsibilityState.CURRENT_CONFIRMED_OWNER
    assert not cross_platform_policy_is_qualification_module(row.semantic_owner_path)
    assert row.semantic_owner_path == COMPAT_X_CROSS_PLATFORM_POLICY_AUTHORITY
    assert row.composition_owner_path == COMPAT_X_QUALIFICATION_ENFORCEMENT_ROOT
    assert frz_cmp_02_pass_candidate() is True


def test_cx_r1_adversarial_a_public_stable_missing_version_fails() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import _synthetic_public_stable_record

    record = _synthetic_public_stable_record("")
    clf = synthetic_public_stable_missing_version()
    assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
    assert clf.version_identity_compliance == VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
    assert clf.version_identity_scheme == VersionIdentityScheme.UNCLASSIFIED
    assert classification_r1_qualification_passes(clf, record) is False


def test_cx_r1_r2_adversarial_h_public_missing_no_deferral_fails() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import _synthetic_public_stable_record

    record = _synthetic_public_stable_record("")
    clf = synthetic_public_stable_missing_version()
    assert classification_r1_qualification_passes(clf, record) is False


def test_cx_r1_r2_adversarial_i_public_missing_with_r2_deferral_still_fails_r1() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import _synthetic_public_stable_record

    record = _synthetic_public_stable_record("")
    identity = compute_version_identity_compliance(
        record,
        VersionObligation.EXPLICIT_VERSION_REQUIRED,
        LaterStageDeferral.R2_PERSISTED_MIGRATION,
        VersionAuthorityDisposition.DOMAIN_VERSION_OWNER,
        R1ComplianceState.COMPLIANT,
    )
    assert identity == VersionIdentityCompliance.BLOCKED_LATER_STAGE
    clf = synthetic_public_stable_missing_version()
    assert classification_r1_qualification_passes(clf, record) is False


def test_cx_r1_r2_adversarial_j_public_version_present_r2_deferral_passes_r1_identity() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import _synthetic_public_stable_record

    record = _synthetic_public_stable_record("foo.v1")
    identity = compute_version_identity_compliance(
        record,
        VersionObligation.EXPLICIT_VERSION_REQUIRED,
        LaterStageDeferral.R2_PERSISTED_MIGRATION,
        VersionAuthorityDisposition.DOMAIN_VERSION_OWNER,
        R1ComplianceState.COMPLIANT,
    )
    assert identity == VersionIdentityCompliance.VERSION_PRESENT_AND_OWNED
    clf = synthetic_public_stable_explicit_version()
    assert classification_r1_qualification_passes(clf, record) is True
    assert clf.later_stage_deferral in {LaterStageDeferral.NOT_APPLICABLE, LaterStageDeferral.R2_PERSISTED_MIGRATION}


def test_cx_r1_r3_adversarial_k_public_inherited_resolved_passes() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import (
        _synthetic_public_stable_inherited_envelope_record,
        synthetic_public_stable_inherited_envelope_resolved,
    )

    record = _synthetic_public_stable_inherited_envelope_record(
        compatibility_policy_owner="intergrax/contracts/migrations/registry.py",
        current_version="synthetic.envelope.v1",
    )
    clf = synthetic_public_stable_inherited_envelope_resolved()
    assert clf.version_obligation == VersionObligation.VERSION_INHERITED_FROM_CANONICAL_ENVELOPE
    assert clf.version_identity_compliance == VersionIdentityCompliance.VERSION_INHERITED_AND_RESOLVED
    assert classification_r1_qualification_passes(clf, record) is True


def test_cx_r1_r3_adversarial_l_public_inherited_unresolved_fails() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import (
        _synthetic_public_stable_inherited_envelope_record,
        synthetic_public_stable_inherited_envelope_unresolved,
    )

    record = _synthetic_public_stable_inherited_envelope_record(
        compatibility_policy_owner="synthetic/unowned_envelope.py",
        current_version="",
    )
    clf = synthetic_public_stable_inherited_envelope_unresolved()
    assert clf.version_obligation == VersionObligation.VERSION_INHERITED_FROM_CANONICAL_ENVELOPE
    assert clf.version_identity_compliance == VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING
    assert classification_r1_qualification_passes(clf, record) is False


def test_cx_r1_r2_adversarial_m_proof_receipt_production_surface() -> None:
    proof_rows = [
        r
        for r in COMPAT_X_INVENTORY
        if r.semantic_identity
        == "class.field:intergrax/proofs/receipts/contracts.py:ProofReceipt:schema_version"
        and r.owner_module_path == "intergrax/proofs/receipts/contracts.py"
    ]
    assert proof_rows, "ProofReceipt.schema_version must appear in closed-world inventory"
    row = proof_rows[0]
    assert "ProofReceipt.schema_version" in row.version_source
    assert "PROOF_RECEIPT_SCHEMA_VERSION" in row.version_source
    clf = classify_inventory_surface(row)
    assert row.current_version == "intergrax.proof_receipt.v1"
    assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
    assert clf.version_identity_compliance == VersionIdentityCompliance.VERSION_PRESENT_AND_OWNED
    assert clf.version_identity_scheme == VersionIdentityScheme.SCHEMA_ID_GENERATION
    assert classification_r1_qualification_passes(clf, row) is True
    const_rows = [
        r
        for r in COMPAT_X_INVENTORY
        if r.owner_module_path == "intergrax/proofs/receipts/contracts.py"
        and "PROOF_RECEIPT_SCHEMA_VERSION" in r.surface_id
    ]
    assert const_rows
    assert public_stable_noncompliant_version_surfaces() == ()


def test_cx_r1_adversarial_a2_public_stable_explicit_version_passes() -> None:
    from tests.qualification.compat_x._compat_x_versioning_gates import _synthetic_public_stable_record

    record = _synthetic_public_stable_record("foo.v1")
    clf = synthetic_public_stable_explicit_version()
    assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
    assert clf.version_identity_compliance == VersionIdentityCompliance.VERSION_PRESENT_AND_OWNED
    assert clf.version_identity_scheme == VersionIdentityScheme.SCHEMA_ID_GENERATION
    assert classification_r1_qualification_passes(clf, record) is True


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
        assert clf.version_identity_compliance == VersionIdentityCompliance.VERSION_REQUIRED_BUT_MISSING


def test_cx_r1_adversarial_c_additive_with_family_policy_passes() -> None:
    family = FamilyCompatibilityPolicy(
        family_id="synthetic.tolerant.reader",
        additive_read_compatibility_supported=True,
        unknown_fields_accepted=True,
        explicit_decoder_or_reader_evidence=True,
        owner_evidence_reference="synthetic/evidence/tolerant_reader.md",
    )
    verdict = evaluate_change_classification(
        CompatibilityChangeClass.ADDITIVE_BACKWARD_COMPATIBLE,
        family_policy=family,
    )
    assert verdict.qualification_passes is True


def test_cx_r1_adversarial_c2_additive_without_family_policy_fails_closed() -> None:
    verdict = evaluate_change_classification(CompatibilityChangeClass.ADDITIVE_BACKWARD_COMPATIBLE)
    assert verdict.qualification_passes is False


def test_cx_r1_adversarial_c3_strict_family_additive_fails_closed() -> None:
    strict = FamilyCompatibilityPolicy(
        family_id="synthetic.extra_forbid",
        additive_read_compatibility_supported=False,
        unknown_fields_accepted=False,
        explicit_decoder_or_reader_evidence=False,
        owner_evidence_reference="synthetic/evidence/forbid_extra.md",
    )
    verdict = evaluate_change_classification(
        CompatibilityChangeClass.ADDITIVE_BACKWARD_COMPATIBLE,
        family_policy=strict,
    )
    assert verdict.qualification_passes is False


def test_cx_r1_adversarial_c4_tolerant_adapter_family_passes_with_evidence() -> None:
    tolerant = FamilyCompatibilityPolicy(
        family_id="synthetic.adapter.tolerant",
        additive_read_compatibility_supported=True,
        unknown_fields_accepted=True,
        explicit_decoder_or_reader_evidence=True,
        owner_evidence_reference="intergrax/compat/langchain/documents.py",
    )
    verdict = evaluate_change_classification(
        CompatibilityChangeClass.ADDITIVE_BACKWARD_COMPATIBLE,
        family_policy=tolerant,
    )
    assert verdict.qualification_passes is True


def test_cx_r1_adversarial_d_breaking_structural_no_bump_fails() -> None:
    verdict = evaluate_declared_change_with_bump(CompatibilityChangeClass.BREAKING_STRUCTURAL, False)
    assert verdict.qualification_passes is False


def test_cx_r1_adversarial_e_semantic_change_same_shape_no_bump_fails() -> None:
    verdict = evaluate_declared_change_with_bump(CompatibilityChangeClass.BREAKING_SEMANTIC, False)
    assert verdict.qualification_passes is False
    assert verdict.requires_new_version_identity is True


def test_cx_r1_adversarial_f_unknown_compatibility_impact_fails_closed() -> None:
    verdict = evaluate_change_classification(CompatibilityChangeClass.UNKNOWN)
    assert verdict.qualification_passes is False


def test_cx_r1_adversarial_g_entity_version_int_not_schema_version() -> None:
    assert classify_version_field_role("version") == VersionFieldRole.BUSINESS_ENTITY_REVISION
    clf = classify_inventory_surface(
        next(r for r in COMPAT_X_INVENTORY if r.surface_id.endswith(":platform_version"))
    )
    assert clf.version_field_role == VersionFieldRole.NON_SCHEMA_DOMAIN_VERSION
    assert clf.version_obligation == VersionObligation.NOT_APPLICABLE


def test_cx_r1_adversarial_h_optimistic_concurrency_not_schema_version() -> None:
    assert classify_version_field_role("_version") == VersionFieldRole.OPTIMISTIC_CONCURRENCY
    role = classify_version_field_role("state_version")
    assert role == VersionFieldRole.OPTIMISTIC_CONCURRENCY


def test_cx_r1_adversarial_i_explicit_schema_version_maps_to_domain_owner() -> None:
    contract = next(r for r in COMPAT_X_INVENTORY if r.surface_id.startswith("semantic.registry.contract:"))
    clf = classify_inventory_surface(contract)
    assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
    assert clf.authority.domain_version_owner_path is not None
    assert "contracts/migrations/registry.py" in clf.authority.domain_version_owner_path
    assert clf.authority.cross_platform_policy_authority == COMPAT_X_CROSS_PLATFORM_POLICY_AUTHORITY
    assert clf.authority.domain_version_disposition == VersionAuthorityDisposition.DOMAIN_VERSION_OWNER


def test_cx_r1_adversarial_j_conflicting_registry_versions_fail() -> None:
    overlap = analyze_registry_overlap()
    assert overlap.version_conflicts == ()
    assert registry_declared_versions_match() == ()


def test_cx_r1_adversarial_k_shim_claiming_canonical_version_fails() -> None:
    assert (
        evaluate_shim_canonical_authority(
            ShimCanonicalAuthorityProbe(
                claims_canonical_current_version=True,
                adapter_supported_old_version="v1",
                canonical_owner_disposition=VersionAuthorityDisposition.DOMAIN_VERSION_OWNER,
                canonical_version_owner_path="intergrax/contracts/migrations/registry.py",
            )
        )
        is False
    )


def test_cx_r1_adversarial_l_shim_supports_old_version_canonical_owner_external_passes() -> None:
    assert evaluate_shim_canonical_authority(
        ShimCanonicalAuthorityProbe(
            claims_canonical_current_version=False,
            adapter_supported_old_version="v1",
            canonical_owner_disposition=VersionAuthorityDisposition.DOMAIN_VERSION_OWNER,
            canonical_version_owner_path="intergrax/contracts/migrations/registry.py",
        )
    )


def test_cx_r1_adversarial_m_qualification_not_semantic_policy_owner() -> None:
    row = next(r for r in COMPAT_X_OWNER_MATRIX if r.concern == "versioning_policy")
    assert not cross_platform_policy_is_qualification_module(row.semantic_owner_path)


def test_cx_r1_adversarial_n_explicit_required_unknown_scheme_unclassified() -> None:
    clf = synthetic_public_stable_missing_version()
    assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
    assert clf.version_identity_scheme == VersionIdentityScheme.UNCLASSIFIED


def test_cx_r1_adversarial_o_event_surface_has_r1_obligation_deferred_r3() -> None:
    event_rows = [r for r in COMPAT_X_INVENTORY if ExposureFacet.EVENT_SCHEMA in r.exposure_facets]
    assert event_rows
    for row in event_rows:
        clf = classify_inventory_surface(row)
        assert clf.version_obligation == VersionObligation.EXPLICIT_VERSION_REQUIRED
        if row.surface_id.startswith("semantic.registry.event:"):
            assert clf.later_stage_deferral == LaterStageDeferral.R3_EVENT_EVOLUTION


def test_cx_r1_adversarial_p_plugin_surface_has_r1_obligation_deferred_r4() -> None:
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
    assert inventory_shim_canonical_authority_violations() == ()
    shims = [r for r in COMPAT_X_INVENTORY if ExposureFacet.COMPATIBILITY_ADAPTER in r.exposure_facets]
    for row in shims:
        clf = classify_inventory_surface(row)
        assert clf.version_obligation == VersionObligation.INTERNAL_NON_VERSIONED_ALLOWED
        assert clf.authority.domain_version_disposition == VersionAuthorityDisposition.NO_VERSION_AUTHORITY
        assert clf.authority.domain_version_owner_path is None
        assert clf.authority.cross_platform_policy_authority == COMPAT_X_CROSS_PLATFORM_POLICY_AUTHORITY


def test_cx_r1_present_version_has_classified_scheme() -> None:
    for clf in COMPAT_X_R1_CLASSIFICATIONS:
        if clf.version_identity_compliance == VersionIdentityCompliance.VERSION_PRESENT_AND_OWNED:
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
