# © Artur Czarnecki. All rights reserved.

"""GR-12-A3 — core control-plane surface execution qualification gates."""

from __future__ import annotations

import pytest

from intergrax.agent_distribution.control_plane_governance import (
    MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
    MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
    MUTATION_TYPE_BIND_AGENT,
    MUTATION_TYPE_BUILD_RUNTIME_REVISION,
    MUTATION_TYPE_COMPLETE_DRAIN,
    MUTATION_TYPE_DISABLE_BINDING,
    MUTATION_TYPE_ENABLE_BINDING,
    MUTATION_TYPE_INSTALL_AGENT,
    MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE,
    MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
    MUTATION_TYPE_UPDATE_BINDING_CONFIG,
)
from tests.qualification.governance.gr12.a3_path_qualifications import (
    GR12_A3_AD_PATH_TO_MUTATION_TYPE,
    GR12_A3_CORE_PATH_PROOFS,
    GR12_A3_QUALIFIED_PATH_IDS,
    gr12_a3_proof_bundle,
)
from tests.qualification.governance.gr12.catalog import (
    GR12_A3_NEXT_REMEDIATION,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_gr12_a3_path_semantic_integrity,
    assert_proof_nodes_registered,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_OVERALL_FORMAL_CLOSURE,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_AD_MUTATION_TYPES = frozenset(
    {
        MUTATION_TYPE_ACTIVATE_RUNTIME_REVISION,
        MUTATION_TYPE_ROLLBACK_RUNTIME_REVISION,
        MUTATION_TYPE_INSTALL_AGENT,
        MUTATION_TYPE_BIND_AGENT,
        MUTATION_TYPE_UPDATE_BINDING_CONFIG,
        MUTATION_TYPE_ENABLE_BINDING,
        MUTATION_TYPE_DISABLE_BINDING,
        MUTATION_TYPE_BUILD_RUNTIME_REVISION,
        MUTATION_TYPE_ADMIT_RUNTIME_REVISION,
        MUTATION_TYPE_COMPLETE_DRAIN,
        MUTATION_TYPE_MARK_POST_CUTOVER_FAILURE,
    }
)


def test_gr12_a3_catalog_qualified_paths_match_ssot() -> None:
    qualified_in_catalog = {
        row.path_id
        for row in GR12_CONTROL_PLANE_SURFACES
        if row.coverage is Gr12CoverageStatus.QUALIFIED
    }
    assert set(GR12_A3_QUALIFIED_PATH_IDS).issubset(qualified_in_catalog)
    assert qualified_in_catalog - set(GR12_A3_QUALIFIED_PATH_IDS) == {
        "CP-PLUGIN-CATALOG-HOT-RELOAD",
    }


def test_gr12_a3_qualified_rows_carry_execution_proof_reference() -> None:
    for row in GR12_CONTROL_PLANE_SURFACES:
        if row.coverage is Gr12CoverageStatus.QUALIFIED:
            assert row.qualification_proof.strip()
            if row.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD":
                continue
            bundle = gr12_a3_proof_bundle(row.path_id)
            assert row.qualification_proof == bundle.primary_proof
        elif row.applicability is Gr12Applicability.APPLICABLE:
            assert row.coverage is not Gr12CoverageStatus.QUALIFIED


def test_gr12_a3_proof_nodes_exist() -> None:
    nodes: list[str] = []
    for bundle in GR12_A3_CORE_PATH_PROOFS:
        nodes.extend(bundle.all_proof_nodes())
    assert_proof_nodes_registered(tuple(nodes))


def test_gr12_a3_ad_mutation_inventory_exhaustive() -> None:
    assert set(GR12_A3_AD_PATH_TO_MUTATION_TYPE.values()) == _AD_MUTATION_TYPES
    assert set(GR12_A3_AD_PATH_TO_MUTATION_TYPE) == {
        row.path_id for row in GR12_CONTROL_PLANE_SURFACES if row.path_id.startswith("CP-AD-")
    }


def test_gr12_a3_matrix_mandatory_invariants_per_qualified_path() -> None:
    for bundle in GR12_A3_CORE_PATH_PROOFS:
        assert_gr12_a3_path_semantic_integrity(bundle)


def test_gr12_a3_task_control_paths_no_longer_gap() -> None:
    for path_id in ("CP-TASK-CANCEL", "CP-TASK-RESUME", "CP-TASK-AUTONOMY"):
        row = next(r for r in GR12_CONTROL_PLANE_SURFACES if r.path_id == path_id)
        assert row.coverage is Gr12CoverageStatus.QUALIFIED


def test_gr12_a3_gr10_remains_final_closed() -> None:
    assert "FINAL CLOSED" in GR10_OVERALL_FORMAL_CLOSURE.status


def test_gr12_a3_next_bounded_task_is_a4() -> None:
    assert "GR-12-A4" in GR12_A3_NEXT_REMEDIATION.task_name
