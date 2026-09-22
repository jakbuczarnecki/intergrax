# © Artur Czarnecki. All rights reserved.

"""GR-12-A4 — residual catalog, vector, and memory control-plane classification gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gr12.a4_residual_path_classifications import (
    GR12_A4_CATALOG_DECISION,
    GR12_A4_MEMORY_DECISION,
    GR12_A4_RESIDUAL_INVENTORY,
    GR12_A4_RESIDUAL_PATH_IDS,
    GR12_A4_VECTOR_DECISION,
    Gr12CatalogRevisionSemantics,
    Gr12MemoryCla04Compatibility,
    Gr12OperatorApiExposure,
)
from tests.qualification.governance.gr12.catalog import (
    GR12_A4_NEXT_REMEDIATION,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.a3_path_qualifications import (
    GR12_A3_QUALIFIED_PATH_IDS,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_OVERALL_FORMAL_CLOSURE,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_WIRING = _REPO_ROOT / "intergrax/applications/_shared/environment_wiring.py"
_CATALOG_WIRING = _REPO_ROOT / "intergrax/applications/_shared/catalog_hot_reload_wiring.py"


def _catalog_row(path_id: str):
    return next(row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == path_id)


def test_gr12_a4_residual_paths_classified() -> None:
    assert tuple(row.path_id for row in GR12_A4_RESIDUAL_INVENTORY) == GR12_A4_RESIDUAL_PATH_IDS
    for inv in GR12_A4_RESIDUAL_INVENTORY:
        row = _catalog_row(inv.path_id)
        assert row.coverage is inv.coverage
        assert inv.consequential
        if inv.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD":
            assert row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED
            assert row.applicability is Gr12Applicability.APPLICABLE
            assert row.qualification_proof.strip()
        else:
            assert row.coverage is Gr12CoverageStatus.ARCHITECTURE_DECISION_REQUIRED
            assert row.applicability is Gr12Applicability.REQUIRES_ARCHITECTURE_DECISION
            assert not row.qualification_proof.strip()


def test_gr12_a4_catalog_residual_wired_not_qualified() -> None:
    for path_id in GR12_A4_RESIDUAL_PATH_IDS:
        row = _catalog_row(path_id)
        if path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD":
            assert row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED
        else:
            assert row.coverage is not Gr12CoverageStatus.QUALIFIED


def test_gr12_a4_a3_qualified_paths_unchanged() -> None:
    qualified = {
        row.path_id
        for row in GR12_CONTROL_PLANE_SURFACES
        if row.coverage is Gr12CoverageStatus.QUALIFIED
    }
    assert set(GR12_A3_QUALIFIED_PATH_IDS).issubset(qualified)
    assert "CP-PLUGIN-CATALOG-HOT-RELOAD" not in qualified


def test_gr12_a4_catalog_hot_reload_not_host_compose_wired() -> None:
    assert _ENV_WIRING.is_file()
    env_source = _ENV_WIRING.read_text(encoding="utf-8")
    assert "catalog_hot_reload" not in env_source
    assert GR12_A4_CATALOG_DECISION.host_compose_wired is False
    assert (
        GR12_A4_CATALOG_DECISION.revision_semantics
        is Gr12CatalogRevisionSemantics.PRESENT
    )
    assert GR12_A4_CATALOG_DECISION.stale_cas_possible is True


def test_gr12_a4_catalog_wiring_exposes_governed_service() -> None:
    source = _CATALOG_WIRING.read_text(encoding="utf-8")
    assert "CatalogHotReloadService" in source
    assert "reload_integration_catalog" not in source


def test_gr12_a4_vector_port_neutral_entrypoint_in_catalog() -> None:
    row = _catalog_row("CP-VECTOR-INDEX-ADMIN")
    assert "integrations.contracts.vector_index_administration" in row.production_entrypoint
    assert GR12_A4_VECTOR_DECISION.cla04_mapping_decision_required
    assert not GR12_A4_VECTOR_DECISION.destructive_ops_on_port


def test_gr12_a4_memory_classification_option_c() -> None:
    assert (
        GR12_A4_MEMORY_DECISION.cla04_compatibility
        is Gr12MemoryCla04Compatibility.C_SEPARATE_POLICY_EVIDENCE_CONTRACT
    )
    row = _catalog_row("CP-MEM-SPECIALIZED-MUTATION")
    assert "MemoryGovernanceEvaluationRequest" in row.current_guard


def test_gr12_a4_operator_exposure_honesty() -> None:
    by_id = {row.path_id: row for row in GR12_A4_RESIDUAL_INVENTORY}
    assert (
        by_id["CP-PLUGIN-CATALOG-HOT-RELOAD"].operator_exposure
        is Gr12OperatorApiExposure.OPERATOR_REQUEST_PATH
    )
    assert (
        by_id["CP-VECTOR-INDEX-ADMIN"].operator_exposure
        is Gr12OperatorApiExposure.NOT_CURRENTLY_EXPOSED
    )
    assert (
        by_id["CP-MEM-SPECIALIZED-MUTATION"].operator_exposure
        is Gr12OperatorApiExposure.NOT_CURRENTLY_EXPOSED
    )


def test_gr12_a4_gr10_remains_final_closed() -> None:
    assert "FINAL CLOSED" in GR10_OVERALL_FORMAL_CLOSURE.status


def test_gr12_a4_next_bounded_task_is_catalog_r1_r1() -> None:
    assert "GR-12-A4-R1-R1" in GR12_A4_NEXT_REMEDIATION.task_name
    blocker = GR12_A4_NEXT_REMEDIATION.exact_blocker.lower()
    assert "revision" in blocker and "authoritative" in blocker
    assert "aba" in blocker
    assert "requestidentity" in blocker.replace(" ", "") or "request identity" in blocker


def test_gr12_a4_catalog_ssot_wired_not_qualified() -> None:
    row = _catalog_row("CP-PLUGIN-CATALOG-HOT-RELOAD")
    assert row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED
