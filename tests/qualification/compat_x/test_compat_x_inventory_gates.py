# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-P0-R1 closed-world inventory gates and adversarial probes."""

from __future__ import annotations

import pytest

from intergrax.runtime.events.payload_registry import UnknownPayloadSchemaError, validate_payload_envelope
from intergrax.runtime.schema.registry import validate_schema_version

from tests.qualification.compat_x._compat_x_classifiers import (
    classify_migration_module,
    classify_shim_module,
    langchain_documents_shim_evidence,
    sanctioned_migration_classes,
)
from tests.qualification.compat_x._compat_x_closed_world import build_closed_world_report
from tests.qualification.compat_x._compat_x_discovery import (
    build_shim_authority_scope_reconciliation,
    closed_world_parity_holds,
    discover_all_compat_surface_ids,
    discover_class_field_versions_from_source,
    discover_migration_mechanism_surfaces,
    discover_shim_surfaces,
)
from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY, compat_x_inventory
from tests.qualification.compat_x._compat_x_owner_discovery import COMPAT_X_OWNER_MATRIX, owner_matrix_paths_exist
from tests.qualification.compat_x._compat_x_registry_analysis import analyze_registry_overlap
from tests.qualification.compat_x._compat_x_synthetic import (
    SYNTHETIC_DECODE_NORMALIZATION_SOURCE,
    SYNTHETIC_MODULE_PATH_MIGRATION,
    SYNTHETIC_MODULE_PATH_PARALLEL,
    SYNTHETIC_MODULE_PATH_EXTERNAL_LEGACY_REGISTRY,
    SYNTHETIC_MODULE_PATH_LEGACY_RUNTIME_ADAPTER,
    SYNTHETIC_MODULE_PATH_LEGACY_TOP_LEVEL_AUTHORIZER,
    SYNTHETIC_MODULE_PATH_LEGACY_TOP_LEVEL_EXECUTOR,
    SYNTHETIC_MODULE_PATH_LEGACY_TRANSLATION_ADAPTER,
    SYNTHETIC_MODULE_PATH_PARALLEL_PROBES,
    SYNTHETIC_MODULE_PATH_PERSISTED,
    SYNTHETIC_MODULE_PATH_PERSISTED_CONTRACT_UNVERSIONED,
    SYNTHETIC_MODULE_PATH_PERSISTED_CONTRACT_VERSIONED,
    SYNTHETIC_MODULE_PATH_PUBLIC,
    SYNTHETIC_PARALLEL_AUTHORITY_SOURCE,
    SYNTHETIC_PARALLEL_AUTHORIZER_SOURCE,
    SYNTHETIC_PARALLEL_EXECUTOR_SOURCE,
    SYNTHETIC_PARALLEL_LEGACY_REGISTRY_SOURCE,
    SYNTHETIC_PARALLEL_PROVIDER_SELECTOR_SOURCE,
    SYNTHETIC_PARALLEL_REGISTRY_SOURCE,
    SYNTHETIC_LEGACY_TRANSLATION_OUTSIDE_COMPAT_SOURCE,
    SYNTHETIC_TOP_LEVEL_AUTHORIZE_LEGACY_SOURCE,
    SYNTHETIC_TOP_LEVEL_EXECUTE_LEGACY_SOURCE,
    SYNTHETIC_PERSISTED_CONTRACT_VERSION_REMOVED_SOURCE,
    SYNTHETIC_PERSISTED_CONTRACT_VERSIONED_SOURCE,
    SYNTHETIC_PERSISTED_WITHOUT_VERSION_SOURCE,
    SYNTHETIC_PUBLIC_CONTRACT_SOURCE,
    SYNTHETIC_TRANSLATION_ONLY_SOURCE,
    SYNTHETIC_UNSANCTIONED_MIGRATION_SOURCE,
)
from tests.qualification.compat_x._compat_x_types import (
    EvolutionState,
    ExposureFacet,
    FrzCmpCandidate,
    MigrationMechanismClass,
    OwnerResponsibilityState,
    ShimClass,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]


def test_cx_p0_r1_q01_closed_world_parity() -> None:
    report = build_closed_world_report()
    assert report.unclassified_candidate_ids == frozenset()
    assert len(report.resolved_candidate_ids) == len(report.raw_candidates)
    assert closed_world_parity_holds()


def test_cx_p0_r1_q02_inventory_matches_semantic_discovery() -> None:
    discovered = discover_all_compat_surface_ids()
    inventoried = frozenset(row.surface_id for row in COMPAT_X_INVENTORY)
    assert discovered == inventoried
    assert len(discovered) >= 50


def test_cx_p0_r1_q03_inventory_unique_ids_and_semantic_identity() -> None:
    ids = [row.surface_id for row in COMPAT_X_INVENTORY]
    assert len(ids) == len(set(ids))
    for row in COMPAT_X_INVENTORY:
        assert row.semantic_identity
        assert row.evidence_paths
        assert row.exposure_facets
        assert EvolutionState.UNCLASSIFIED not in row.evolution_states


def test_cx_p0_r1_q04_owner_matrix_evidence_paths() -> None:
    assert len(COMPAT_X_OWNER_MATRIX) >= 9
    assert owner_matrix_paths_exist() == ()
    fragmented = [
        r for r in COMPAT_X_OWNER_MATRIX if r.responsibility_state == OwnerResponsibilityState.FRAGMENTED_UNOWNED
    ]
    assert fragmented


def test_cx_p0_r1_q05_no_parallel_authority_in_production_inventory() -> None:
    parallel = [
        row
        for row in COMPAT_X_INVENTORY
        if row.shim_class == ShimClass.PARALLEL_AUTHORITY
        or EvolutionState.COMPATIBILITY_SHIM_PARALLEL_AUTHORITY in row.evolution_states
    ]
    assert parallel == []


def test_cx_p0_r1_frz_cmp_01_pass_candidate_when_parity_holds() -> None:
    report = build_closed_world_report()
    assert report.unclassified_candidate_ids == frozenset()
    assert discover_all_compat_surface_ids() == frozenset(r.surface_id for r in compat_x_inventory())
    assert FrzCmpCandidate.PASS_CANDIDATE.value == "PASS_CANDIDATE"


def test_cx_p0_frz_cmp_02_versioning_policy_r1_defined() -> None:
    missing_policy = [
        row.surface_id
        for row in COMPAT_X_INVENTORY
        if EvolutionState.VERSIONED_POLICY_MISSING in row.evolution_states
    ]
    assert missing_policy == []
    defined = [
        row.surface_id
        for row in COMPAT_X_INVENTORY
        if EvolutionState.VERSIONED_AND_POLICY_DEFINED in row.evolution_states
    ]
    assert len(defined) == len(COMPAT_X_INVENTORY)


def test_cx_p0_r1_adversarial_01_class_field_without_module_constant_discovered() -> None:
    found = discover_class_field_versions_from_source(SYNTHETIC_MODULE_PATH_PUBLIC, SYNTHETIC_PUBLIC_CONTRACT_SOURCE)
    assert any("SyntheticPublicContract" in c.discovered_signal for c in found)
    assert any(c.semantic_identity == "schema.literal:synthetic.v1" for c in found)


def test_cx_p0_r1_adversarial_02_omitted_public_contract_breaks_parity() -> None:
    from tests.qualification.compat_x._compat_x_closed_world import (
        build_closed_world_report_with_extra_candidates,
        discover_candidates_from_source,
    )

    extra = tuple(discover_candidates_from_source(SYNTHETIC_MODULE_PATH_PUBLIC, SYNTHETIC_PUBLIC_CONTRACT_SOURCE))
    assert extra
    identity = "schema.literal:synthetic.v1"
    with_extra = build_closed_world_report_with_extra_candidates(extra)
    assert identity in {s.semantic_identity for s in with_extra.semantic_surfaces}
    base = build_closed_world_report()
    assert identity not in {s.semantic_identity for s in base.semantic_surfaces}


def test_cx_p0_r1_adversarial_03_persisted_without_version_flagged_by_discovery() -> None:
    from tests.qualification.compat_x._compat_x_closed_world import discover_candidates_from_source

    candidates = discover_candidates_from_source(
        SYNTHETIC_MODULE_PATH_PERSISTED, SYNTHETIC_PERSISTED_WITHOUT_VERSION_SOURCE
    )
    defects = [c for c in candidates if c.discovery_kind == "defect.persisted_without_version"]
    assert defects
    assert defects[0].semantic_identity.startswith("persisted.without_version:")


def test_cx_p0_r1_adversarial_04_unknown_runtime_schema_version_rejected() -> None:
    assert validate_schema_version("runtime_event", "runtime_event.v999") is False


def test_cx_p0_r1_adversarial_05_unknown_event_payload_rejected() -> None:
    with pytest.raises(UnknownPayloadSchemaError):
        validate_payload_envelope(
            {"payload_schema_id": "synthetic.unknown.event.probe", "data": {}}
        )


def test_cx_p0_r1_adversarial_06_unsanctioned_migration_mechanism_detected() -> None:
    cls = classify_migration_module(SYNTHETIC_MODULE_PATH_MIGRATION, SYNTHETIC_UNSANCTIONED_MIGRATION_SOURCE)
    assert cls == MigrationMechanismClass.UNSANCTIONED_MIGRATION_MECHANISM
    assert cls not in sanctioned_migration_classes()
    production = [
        r for r in COMPAT_X_INVENTORY if r.migration_class == MigrationMechanismClass.UNSANCTIONED_MIGRATION_MECHANISM
    ]
    assert production == []


def test_cx_p0_r1_adversarial_07_shim_discovered_without_path_list_edit() -> None:
    shims = discover_shim_surfaces()
    paths = {s.owner_module_path for s in shims}
    assert "intergrax/compat/langchain/documents.py" in paths
    assert len(shims) >= 1


def test_cx_p0_r1_adversarial_08_parallel_authority_classifier_on_synthetic() -> None:
    shim = classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL, SYNTHETIC_PARALLEL_AUTHORITY_SOURCE)
    assert shim == ShimClass.PARALLEL_AUTHORITY


def test_cx_p0_r1_adversarial_09_langchain_shim_translation_evidence() -> None:
    evidence = langchain_documents_shim_evidence()
    assert any("from_langchain_document" in item for item in evidence)
    assert all("resolve_provider" not in item for item in evidence)
    row = next(
        r
        for r in COMPAT_X_INVENTORY
        if r.owner_module_path == "intergrax/compat/langchain/documents.py" and r.shim_class != ShimClass.NOT_APPLICABLE
    )
    assert row.shim_class == ShimClass.READ_COMPATIBILITY_ONLY


def test_cx_p0_r1_adversarial_10_registry_version_conflict_gate() -> None:
    matrix = analyze_registry_overlap()
    assert matrix.version_conflicts == ()


def test_cx_p0_r1_adversarial_11_tenant_invariant_reuse_extcomp() -> None:
    from tests.qualification.external_contract_compatibility.test_external_contract_compatibility_certification import (  # noqa: PLC0415
        test_cert_22_resolver_tenant_isolation,
    )

    test_cert_22_resolver_tenant_isolation()


def test_cx_p0_r2_adversarial_12_version_removal_regression_probe() -> None:
    from tests.qualification.compat_x._compat_x_closed_world import discover_candidates_from_source
    from tests.qualification.compat_x._compat_x_inventory import _record_from_surface
    from tests.qualification.compat_x._compat_x_closed_world import _defect_surface, _semantic_surface_from_candidate

    versioned = discover_candidates_from_source(
        SYNTHETIC_MODULE_PATH_PERSISTED_CONTRACT_VERSIONED,
        SYNTHETIC_PERSISTED_CONTRACT_VERSIONED_SOURCE,
    )
    version_removed = discover_candidates_from_source(
        SYNTHETIC_MODULE_PATH_PERSISTED_CONTRACT_UNVERSIONED,
        SYNTHETIC_PERSISTED_CONTRACT_VERSION_REMOVED_SOURCE,
    )
    assert any(c.discovery_kind == "class.field.version" for c in versioned)
    assert not any(c.discovery_kind == "defect.persisted_without_version" for c in versioned)
    defects = [c for c in version_removed if c.discovery_kind == "defect.persisted_without_version"]
    assert len(defects) == 1
    assert "PersistedContract" in defects[0].semantic_identity
    defect_surface = _defect_surface(defects[0])
    assert defect_surface is not None
    record = _record_from_surface(defect_surface)
    assert EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION in record.evolution_states
    versioned_surfaces = [
        _semantic_surface_from_candidate(c)
        for c in versioned
        if c.discovery_kind == "class.field.version"
    ]
    assert versioned_surfaces
    versioned_record = _record_from_surface(versioned_surfaces[0])
    assert EvolutionState.PERSISTED_SCHEMA_WITHOUT_VERSION not in versioned_record.evolution_states


def test_cx_p0_r2_parallel_authority_provider_selection_probe() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL_PROBES, SYNTHETIC_PARALLEL_PROVIDER_SELECTOR_SOURCE)
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r2_parallel_authority_registry_ownership_probe() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL_PROBES, SYNTHETIC_PARALLEL_REGISTRY_SOURCE)
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r2_parallel_authority_execution_probe() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL_PROBES, SYNTHETIC_PARALLEL_EXECUTOR_SOURCE)
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r2_parallel_authority_authorization_probe() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL_PROBES, SYNTHETIC_PARALLEL_AUTHORIZER_SOURCE)
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r2_translation_only_sanctioned_probe() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL_PROBES, SYNTHETIC_TRANSLATION_ONLY_SOURCE)
        == ShimClass.TRANSLATION_ONLY
    )


def test_cx_p0_r2_legacy_decode_sanctioned_probe() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_PARALLEL_PROBES, SYNTHETIC_DECODE_NORMALIZATION_SOURCE)
        == ShimClass.TRANSLATION_ONLY
    )


def test_cx_p0_r2_frz_cmp_08_pass_candidate_scope() -> None:
    shims = [r for r in COMPAT_X_INVENTORY if r.shim_class != ShimClass.NOT_APPLICABLE]
    assert shims
    assert all(r.shim_class != ShimClass.PARALLEL_AUTHORITY for r in shims)


def test_cx_p0_r3_shim_authority_scope_reconciliation() -> None:
    reconciliation = build_shim_authority_scope_reconciliation()
    assert reconciliation.total_compatibility_candidates > 0
    assert reconciliation.authority_inspected_compatibility_candidates == reconciliation.total_compatibility_candidates
    assert reconciliation.uninspected_compatibility_candidates == 0
    assert reconciliation.production_parallel_authority_count == 0


def test_cx_p0_r3_adversarial_a_parallel_authority_outside_compat_tree() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_LEGACY_RUNTIME_ADAPTER, SYNTHETIC_PARALLEL_PROVIDER_SELECTOR_SOURCE)
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r3_adversarial_b_external_legacy_registry_parallel_authority() -> None:
    assert (
        classify_shim_module(SYNTHETIC_MODULE_PATH_EXTERNAL_LEGACY_REGISTRY, SYNTHETIC_PARALLEL_LEGACY_REGISTRY_SOURCE)
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r3_adversarial_c_top_level_executor_parallel_authority() -> None:
    assert (
        classify_shim_module(
            SYNTHETIC_MODULE_PATH_LEGACY_TOP_LEVEL_EXECUTOR,
            SYNTHETIC_TOP_LEVEL_EXECUTE_LEGACY_SOURCE,
        )
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r3_adversarial_d_top_level_authorizer_parallel_authority() -> None:
    assert (
        classify_shim_module(
            SYNTHETIC_MODULE_PATH_LEGACY_TOP_LEVEL_AUTHORIZER,
            SYNTHETIC_TOP_LEVEL_AUTHORIZE_LEGACY_SOURCE,
        )
        == ShimClass.PARALLEL_AUTHORITY
    )


def test_cx_p0_r3_adversarial_e_translation_outside_compat_not_parallel_authority() -> None:
    assert (
        classify_shim_module(
            SYNTHETIC_MODULE_PATH_LEGACY_TRANSLATION_ADAPTER,
            SYNTHETIC_LEGACY_TRANSLATION_OUTSIDE_COMPAT_SOURCE,
        )
        == ShimClass.TRANSLATION_ONLY
    )


def test_cx_p0_r3_frz_cmp_08_pass_candidate_scope() -> None:
    reconciliation = build_shim_authority_scope_reconciliation()
    assert reconciliation.uninspected_compatibility_candidates == 0
    assert reconciliation.production_parallel_authority_count == 0
    shims = [r for r in COMPAT_X_INVENTORY if r.shim_class != ShimClass.NOT_APPLICABLE]
    assert shims
    assert all(r.shim_class != ShimClass.PARALLEL_AUTHORITY for r in shims)


def test_cx_p0_r1_migration_mechanism_mechanically_discovered() -> None:
    migration_rows = [r for r in COMPAT_X_INVENTORY if r.migration_class != MigrationMechanismClass.NOT_APPLICABLE]
    assert migration_rows
    assert len(migration_rows) == len(discover_migration_mechanism_surfaces())
    assert not any(r.migration_class == MigrationMechanismClass.UNSANCTIONED_MIGRATION_MECHANISM for r in migration_rows)


def test_cx_p0_r1_inventory_counts_reportable() -> None:
    report = build_closed_world_report()
    rows = COMPAT_X_INVENTORY
    public_stable = sum(1 for r in rows if ExposureFacet.PUBLIC_STABLE in r.exposure_facets)
    persisted = sum(1 for r in rows if ExposureFacet.PERSISTED_SCHEMA in r.exposure_facets)
    events = sum(1 for r in rows if ExposureFacet.EVENT_SCHEMA in r.exposure_facets)
    plugin = sum(1 for r in rows if ExposureFacet.PLUGIN_PROVIDER_CONTRACT in r.exposure_facets)
    shims = sum(1 for r in rows if ExposureFacet.COMPATIBILITY_ADAPTER in r.exposure_facets)
    assert report.raw_candidates
    assert len(rows) <= len(report.raw_candidates)
    assert public_stable >= 5
    assert persisted >= 5
    assert events >= 10
    assert plugin >= 2
    assert shims >= 1


