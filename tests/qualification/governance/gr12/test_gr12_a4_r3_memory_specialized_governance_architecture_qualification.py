# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R3 — specialized memory governance architecture qualification gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gr12.a4_residual_path_classifications import (
    GR12_A4_MEMORY_DECISION,
    GR12_A4_RESIDUAL_INVENTORY,
)
from tests.qualification.governance.gr12.catalog import (
    GR12_A4_R3_MEMORY_ADR_PATH,
    GR12_CONTROL_PLANE_SURFACES,
    Gr12Applicability,
    Gr12CoverageStatus,
)
from tests.qualification.governance.gr12.gr12_a4_r3_memory_architecture_decision import (
    GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION,
    GR12_MEMORY_GOVERNANCE_SERVICE,
    GR12_MEMORY_MUTATION_SURFACES,
    GR12_MEMORY_POLICY_PORT,
    GR12_MEMORY_QUALIFICATION_GATES,
    GR12_MEMORY_REJECTED_ALTERNATIVES,
    Gr12MemoryArchitecturePhase,
    Gr12MemoryAuthorityModel,
    Gr12MemoryGr12Applicability,
    Gr12MemoryMutationContextClass,
)
from tests.qualification.governance.strategy.catalog import GR10_OVERALL_FORMAL_CLOSURE

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _memory_catalog_row():
    return next(
        row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == "CP-MEM-SPECIALIZED-MUTATION"
    )


def test_gr12_a4_r3_memory_architecture_decision_closed_ssot() -> None:
    decision = GR12_A4_R3_MEMORY_ARCHITECTURE_DECISION
    assert decision.architecture_phase is Gr12MemoryArchitecturePhase.ARCHITECTURE_DECISION_CLOSED
    assert decision.authority_model is Gr12MemoryAuthorityModel.SPECIALIZED_DOMAIN_AUTHORITY
    assert decision.memory_native_authority is True
    assert decision.live_operator_surface_exists is False
    assert decision.dual_independent_authority is False
    assert decision.cp_mem_catalog_applicability is Gr12Applicability.NOT_APPLICABLE
    assert decision.cp_mem_catalog_coverage is Gr12CoverageStatus.NOT_APPLICABLE


def test_gr12_a4_r3_memory_mutation_inventory_present() -> None:
    assert len(GR12_MEMORY_MUTATION_SURFACES) >= 5
    live = [
        s
        for s in GR12_MEMORY_MUTATION_SURFACES
        if s.context_class is Gr12MemoryMutationContextClass.LIVE_OPERATOR_CONTROL_PLANE
    ]
    assert len(live) == 1
    assert live[0].gr12_applicability is (
        Gr12MemoryGr12Applicability.APPLICABLE_WHEN_LIVE_OPERATOR_INTRODUCED
    )


def test_gr12_a4_r3_no_control_plane_mutation_in_memory_package() -> None:
    memory_root = _REPO_ROOT / "intergrax" / "memory"
    forbidden = "ControlPlaneMutation"
    for path in memory_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert forbidden not in text


def test_gr12_a4_r3_single_authority_service_documented() -> None:
    service_module = GR12_MEMORY_GOVERNANCE_SERVICE.rsplit(".", 1)[0]
    service_path = _REPO_ROOT / f"{service_module.replace('.', '/')}.py"
    assert service_path.is_file()
    policy_module_path = GR12_MEMORY_POLICY_PORT.rsplit(".", 1)[0]
    policy_module = _REPO_ROOT / f"{policy_module_path.replace('.', '/')}.py"
    assert policy_module.is_file()


def test_gr12_a4_r3_catalog_memory_surface_not_applicable() -> None:
    row = _memory_catalog_row()
    assert row.applicability is Gr12Applicability.NOT_APPLICABLE
    assert row.coverage is Gr12CoverageStatus.NOT_APPLICABLE


def test_gr12_a4_r3_residual_inventory_aligned() -> None:
    inv = next(row for row in GR12_A4_RESIDUAL_INVENTORY if row.path_id == "CP-MEM-SPECIALIZED-MUTATION")
    row = _memory_catalog_row()
    assert inv.coverage is row.coverage


def test_gr12_a4_r3_memory_decision_not_control_plane_compat() -> None:
    from tests.qualification.governance.gr12.a4_residual_path_classifications import (
        Gr12MemoryCla04Compatibility,
    )

    assert (
        GR12_A4_MEMORY_DECISION.cla04_compatibility
        is Gr12MemoryCla04Compatibility.D_NOT_CONTROL_PLANE
    )
    assert not GR12_A4_MEMORY_DECISION.architecture_blocker.strip()


def test_gr12_a4_r3_adr_present_and_covers_decision() -> None:
    adr_path = _REPO_ROOT / GR12_A4_R3_MEMORY_ADR_PATH
    assert adr_path.is_file()
    adr = adr_path.read_text(encoding="utf-8")
    assert "MemoryGovernanceEvaluationRequest" in adr
    assert "MemorySecurityGovernanceService" in adr
    assert "NOT_APPLICABLE" in adr or "not applicable" in adr.lower()
    assert "Rejected" in adr or "rejected" in adr.lower()
    assert "dual" in adr.lower()


def test_gr12_a4_r3_vector_and_catalog_unchanged() -> None:
    vector = next(row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == "CP-VECTOR-INDEX-ADMIN")
    catalog = next(
        row for row in GR12_CONTROL_PLANE_SURFACES if row.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD"
    )
    assert vector.coverage is Gr12CoverageStatus.QUALIFIED
    assert catalog.coverage is Gr12CoverageStatus.QUALIFIED


def test_gr12_a4_r3_gr10_remains_final_closed() -> None:
    assert "FINAL CLOSED" in GR10_OVERALL_FORMAL_CLOSURE.status


def test_gr12_a4_r3_qualification_gates_registered() -> None:
    assert len(GR12_MEMORY_QUALIFICATION_GATES) >= 8


def test_gr12_a4_r3_rejected_dual_authorization_documented() -> None:
    joined = " ".join(GR12_MEMORY_REJECTED_ALTERNATIVES).lower()
    assert "dual" in joined
    assert "cla-04" in joined or "controlplanemutation" in joined.replace(" ", "")
