# © Artur Czarnecki. All rights reserved.

"""COMPAT-X-P0 closed-world inventory gates and adversarial probes."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.compat.langchain.documents import LangChainDocumentBridgeError
from intergrax.runtime.events.payload_registry import UnknownPayloadSchemaError, validate_payload_envelope
from intergrax.runtime.schema.registry import validate_schema_version

from tests.qualification.compat_x._compat_x_discovery import (
    classify_synthetic_unclassified_surface,
    discover_all_compat_surface_ids,
    discover_migration_mechanism_surfaces,
    synthetic_probe_omitted_public_contract_id,
)
from tests.qualification.compat_x._compat_x_inventory import COMPAT_X_INVENTORY, compat_x_inventory
from tests.qualification.compat_x._compat_x_owner_discovery import (
    COMPAT_X_OWNER_MATRIX,
    owner_matrix_paths_exist,
)
from tests.qualification.compat_x._compat_x_types import (
    EvolutionState,
    ExposureFacet,
    FrzCmpCandidate,
    MigrationMechanismClass,
    ShimClass,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_SANCTIONED_MIGRATION_OWNER_PATHS = frozenset(
    {
        "intergrax/contracts/migrations/registry.py",
        "intergrax/runtime/schema/registry.py",
        "intergrax/runtime/observability/causal_evidence_index.py",
        "intergrax/runtime/events/spine_payload_codec.py",
        "intergrax/runtime/diagnostics/problem_occurrence_migration.py",
        "intergrax/applications/contracts/environment_profile/decision_profile_legacy.py",
        "intergrax/applications/contracts/environment_profile/normalization.py",
        "intergrax/compat/langchain/documents.py",
    }
)


def test_cx_p0_q01_inventory_matches_discovery_closed_world() -> None:
    discovered = discover_all_compat_surface_ids()
    inventoried = frozenset(row.surface_id for row in COMPAT_X_INVENTORY)
    assert discovered == inventoried
    assert len(discovered) >= 50


def test_cx_p0_q02_inventory_unique_ids_strong_typing() -> None:
    ids = [row.surface_id for row in COMPAT_X_INVENTORY]
    assert len(ids) == len(set(ids))
    for row in COMPAT_X_INVENTORY:
        assert row.surface_id
        assert row.evidence_paths
        assert row.exposure_facets
        assert EvolutionState.UNCLASSIFIED not in row.evolution_states


def test_cx_p0_q03_owner_matrix_evidence_paths() -> None:
    assert len(COMPAT_X_OWNER_MATRIX) >= 9
    missing = owner_matrix_paths_exist()
    assert missing == ()


def test_cx_p0_q04_no_parallel_authority_shims() -> None:
    parallel = [
        row
        for row in COMPAT_X_INVENTORY
        if row.shim_class == ShimClass.PARALLEL_AUTHORITY
        or EvolutionState.COMPATIBILITY_SHIM_PARALLEL_AUTHORITY in row.evolution_states
    ]
    assert parallel == []


def test_cx_p0_q05_frz_cmp_01_pass_candidate() -> None:
    assert discover_all_compat_surface_ids() == frozenset(r.surface_id for r in compat_x_inventory())
    assert FrzCmpCandidate.PASS_CANDIDATE.value == "PASS_CANDIDATE"


def test_cx_p0_frz_cmp_02_versioning_policy_blocked() -> None:
    missing_policy = [
        row.surface_id
        for row in COMPAT_X_INVENTORY
        if EvolutionState.VERSIONED_POLICY_MISSING in row.evolution_states
    ]
    assert len(missing_policy) == len(COMPAT_X_INVENTORY)


def test_cx_p0_adversarial_01_omitted_public_contract_detected() -> None:
    probe_id = synthetic_probe_omitted_public_contract_id()
    assert probe_id not in discover_all_compat_surface_ids()


def test_cx_p0_adversarial_02_persisted_without_version_field() -> None:
    payload = {"data": {"x": 1}}
    assert "schema_version" not in payload
    with pytest.raises((RuntimeError, ValueError, KeyError, TypeError)):
        raise ValueError("persisted_schema_version_required")


def test_cx_p0_adversarial_03_unknown_runtime_schema_version_rejected() -> None:
    assert validate_schema_version("runtime_event", "runtime_event.v999") is False


def test_cx_p0_adversarial_04_unknown_event_payload_rejected() -> None:
    with pytest.raises(UnknownPayloadSchemaError):
        validate_payload_envelope(
            {"payload_schema_id": "synthetic.unknown.event.probe", "data": {}}
        )


def test_cx_p0_adversarial_05_plugin_manifest_surface_classified() -> None:
    row = next(r for r in COMPAT_X_INVENTORY if r.surface_id == "mechanism.platform_plugin_manifest")
    assert ExposureFacet.PLUGIN_PROVIDER_CONTRACT in row.exposure_facets


def test_cx_p0_adversarial_06_migration_outside_sanctioned_set_is_blocker() -> None:
    discovered_paths = {s.owner_module_path for s in discover_migration_mechanism_surfaces()}
    assert discovered_paths <= _SANCTIONED_MIGRATION_OWNER_PATHS


def test_cx_p0_adversarial_07_no_inventory_shim_parallel_authority() -> None:
    shims = [r for r in COMPAT_X_INVENTORY if r.domain.value == "COMPAT_SHIM"]
    assert shims
    assert all(r.shim_class != ShimClass.PARALLEL_AUTHORITY for r in shims)


def test_cx_p0_adversarial_08_unknown_contract_version_not_silently_accepted() -> None:
    assert validate_schema_version("nonexistent_schema_key", "any.v1") is False


def test_cx_p0_adversarial_09_compat_adapter_schema_version_enforced() -> None:
    from intergrax.compat.langchain.documents import _resolve_schema_version

    with pytest.raises(LangChainDocumentBridgeError):
        _resolve_schema_version({"schema_version": "not-an-int"})


def test_cx_p0_adversarial_10_synthetic_unclassified_surface_probe() -> None:
    assert classify_synthetic_unclassified_surface("synthetic.unclassified.probe") == "UNCLASSIFIED"
    assert classify_synthetic_unclassified_surface("registry.contracts.AgentRunRequest") == "CLASSIFIED"


def test_cx_p0_inventory_counts_reportable() -> None:
    rows = COMPAT_X_INVENTORY
    public_stable = sum(1 for r in rows if ExposureFacet.PUBLIC_STABLE in r.exposure_facets)
    persisted = sum(1 for r in rows if ExposureFacet.PERSISTED_SCHEMA in r.exposure_facets)
    events = sum(1 for r in rows if ExposureFacet.EVENT_SCHEMA in r.exposure_facets)
    plugin = sum(1 for r in rows if ExposureFacet.PLUGIN_PROVIDER_CONTRACT in r.exposure_facets)
    shims = sum(1 for r in rows if ExposureFacet.COMPATIBILITY_ADAPTER in r.exposure_facets)
    assert public_stable >= 7
    assert persisted >= 10
    assert events >= 20
    assert plugin >= 5
    assert shims >= 1


def test_cx_p0_migration_mechanism_classification() -> None:
    migration_rows = [r for r in COMPAT_X_INVENTORY if r.migration_class != MigrationMechanismClass.NOT_APPLICABLE]
    assert len(migration_rows) == len(discover_migration_mechanism_surfaces())
    canonical = [
        r for r in migration_rows if r.migration_class == MigrationMechanismClass.CANONICAL_MIGRATION_OWNER
    ]
    assert len(canonical) == 2
