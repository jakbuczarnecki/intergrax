# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2 — vector administration governance architecture qualification gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gr12.a4_residual_path_classifications import (
    GR12_A4_MEMORY_DECISION,
    GR12_A4_RESIDUAL_INVENTORY,
    GR12_A4_VECTOR_DECISION,
)
from tests.qualification.governance.gr12.catalog import (
    GR12_A4_NEXT_REMEDIATION,
    GR12_A4_R2_QUALIFICATION_PROOF,
    GR12_A4_R2_VECTOR_ADR_PATH,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.gr12_a4_r2_vector_architecture_decision import (
    GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION,
    GR12_VECTOR_CANONICAL_PORT,
    GR12_VECTOR_OPERATION_CLASSIFICATIONS,
    GR12_VECTOR_REJECTED_ALTERNATIVES,
    Gr12VectorIndexOperationClass,
    Gr12VectorPrepareIndexGovernanceDecision,
)
from tests.qualification.governance.strategy.catalog import GR10_OVERALL_FORMAL_CLOSURE

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FORBIDDEN_AUTHORITY_NAMES = (
    "GovernedVectorIndexAdministration",
    "VectorControlPlanePort",
    "QdrantGovernancePort",
)


def _vector_row():
    return next(row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == "CP-VECTOR-INDEX-ADMIN")


def test_gr12_a4_r2_vector_architecture_decision_closed_ssot() -> None:
    decision = GR12_A4_R2_VECTOR_ARCHITECTURE_DECISION
    assert decision.cla04_applicability is Gr12Applicability.APPLICABLE
    assert (
        decision.prepare_index_decision
        is Gr12VectorPrepareIndexGovernanceDecision.OPTION_B_CONDITIONAL_LIVE_OPERATOR_ONLY
    )
    assert decision.live_operator_surface_exists is False
    assert decision.destructive_ops_on_neutral_port is False
    assert decision.canonical_port == GR12_VECTOR_CANONICAL_PORT
    assert decision.next_bounded_task == GR12_A4_NEXT_REMEDIATION.task_name


def test_gr12_a4_r2_vector_catalog_surface_implementation_required() -> None:
    row = _vector_row()
    assert row.applicability is Gr12Applicability.APPLICABLE
    assert row.coverage is Gr12CoverageStatus.IMPLEMENTATION_REQUIRED
    assert row.coverage is not Gr12CoverageStatus.QUALIFIED


def test_gr12_a4_r2_vector_residual_inventory_aligned() -> None:
    inv = next(row for row in GR12_A4_RESIDUAL_INVENTORY if row.path_id == "CP-VECTOR-INDEX-ADMIN")
    row = _vector_row()
    assert inv.coverage is row.coverage
    assert GR12_A4_VECTOR_DECISION.qualification_proof == GR12_A4_R2_QUALIFICATION_PROOF


def test_gr12_a4_r2_single_canonical_vector_admin_port() -> None:
    contract_path = _REPO_ROOT / "intergrax/integrations/contracts/vector_index_administration.py"
    source = contract_path.read_text(encoding="utf-8")
    assert source.count("class VectorIndexAdministration") == 1
    integrations_root = _REPO_ROOT / "intergrax/integrations"
    for path in integrations_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in _FORBIDDEN_AUTHORITY_NAMES:
            assert forbidden not in text


def test_gr12_a4_r2_prepare_index_classified_conditional() -> None:
    prepare = next(
        item for item in GR12_VECTOR_OPERATION_CLASSIFICATIONS if item.operation == "prepare_index"
    )
    assert prepare.operation_class is Gr12VectorIndexOperationClass.CONDITIONAL_MUTATION
    probe = next(item for item in GR12_VECTOR_OPERATION_CLASSIFICATIONS if item.operation == "probe")
    assert probe.operation_class is Gr12VectorIndexOperationClass.READ_ONLY


def test_gr12_a4_r2_no_provider_specific_governance_dispatch_in_decision_ssot() -> None:
    module_path = _REPO_ROOT / "tests/qualification/governance/gr12/gr12_a4_r2_vector_architecture_decision.py"
    source = module_path.read_text(encoding="utf-8").lower()
    assert "if provider ==" not in source
    assert "pinecone" not in source
    assert "weaviate" not in source


def test_gr12_a4_r2_adr_present_and_covers_mapping() -> None:
    adr_path = _REPO_ROOT / GR12_A4_R2_VECTOR_ADR_PATH
    assert adr_path.is_file()
    adr = adr_path.read_text(encoding="utf-8")
    assert "mutation_type" in adr and "vector_index.prepare" in adr
    assert "resource_type" in adr and "vector_index" in adr
    assert "TOCTOU" in adr or "toctou" in adr.lower()
    assert "Rejected" in adr or "rejected" in adr.lower()
    adr_body = adr.split("## Rejected alternatives", maxsplit=1)[0]
    for forbidden in _FORBIDDEN_AUTHORITY_NAMES:
        assert forbidden not in adr_body


def test_gr12_a4_r2_memory_and_catalog_status_unchanged() -> None:
    catalog = next(
        row for row in GR12_A4_RESIDUAL_INVENTORY if row.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD"
    )
    assert catalog.coverage is Gr12CoverageStatus.QUALIFIED
    memory_row = next(
        row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == "CP-MEM-SPECIALIZED-MUTATION"
    )
    assert memory_row.coverage is Gr12CoverageStatus.ARCHITECTURE_DECISION_REQUIRED
    assert GR12_A4_MEMORY_DECISION.architecture_blocker.strip()


def test_gr12_a4_r2_gr10_remains_final_closed() -> None:
    assert "FINAL CLOSED" in GR10_OVERALL_FORMAL_CLOSURE.status


def test_gr12_a4_r2_rejected_alternatives_documented() -> None:
    assert "provider-layer" in " ".join(GR12_VECTOR_REJECTED_ALTERNATIVES).lower()
    assert "second vector admin port" in " ".join(GR12_VECTOR_REJECTED_ALTERNATIVES).lower()
