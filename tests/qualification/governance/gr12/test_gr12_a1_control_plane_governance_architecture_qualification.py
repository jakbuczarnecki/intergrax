# © Artur Czarnecki. All rights reserved.

"""GR-12-A1 — control-plane production surface reconciliation gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.gr12.catalog import (
    GR12_ADR_REQUIRED,
    GR12_A1_NEXT_REMEDIATION,
    GR12_CANONICAL_BOUNDARY_CLASS,
    GR12_CANONICAL_POLICY_PORT,
    GR12_CONTRACT_DECISION,
    GR12_CONTROL_PLANE_SURFACES,
    GR12_CONSEQUENTIAL_MUTATION_DEFINITION,
    GR12_EXECUTION_PLANE_EXCLUSIONS,
    GR12_EXISTING_MECHANISMS,
    Gr12Applicability,
    Gr12CoverageStatus,
    gr12_applicable_surfaces,
    gr12_path_ids,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_OVERALL_FORMAL_CLOSURE,
    GR10_POST_CLOSURE_NEXT_REMEDIATION,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_gr12_a1_path_ids_unique() -> None:
    ids = gr12_path_ids()
    assert len(ids) == len(set(ids))


def test_gr12_a1_no_qualified_without_proof() -> None:
    for row in GR12_CONTROL_PLANE_SURFACES:
        if row.coverage is Gr12CoverageStatus.QUALIFIED:
            assert row.qualification_proof.strip()
        elif row.coverage is Gr12CoverageStatus.WIRED_NOT_QUALIFIED:
            continue
        else:
            assert not row.qualification_proof.strip()


def test_gr12_a1_applicable_surfaces_have_consequential_true() -> None:
    for row in gr12_applicable_surfaces():
        assert row.consequential is True
        assert row.coverage in (
            Gr12CoverageStatus.GAP,
            Gr12CoverageStatus.WIRED_NOT_QUALIFIED,
            Gr12CoverageStatus.QUALIFIED,
            Gr12CoverageStatus.DISCOVERED,
            Gr12CoverageStatus.APPLICABLE,
            Gr12CoverageStatus.IMPLEMENTATION_REQUIRED,
        )


def test_gr12_a1_contract_decision_variant_a() -> None:
    assert GR12_CONTRACT_DECISION == "VARIANT_A_EXISTING_CLA04_CONTRACT_REUSABLE"
    assert GR12_ADR_REQUIRED is False
    assert "ControlPlaneMutationAuthorizationBoundary" in GR12_CANONICAL_BOUNDARY_CLASS
    assert "ControlPlaneMutationPolicyEvaluator" in GR12_CANONICAL_POLICY_PORT


def test_gr12_a1_canonical_contract_modules_exist() -> None:
    assert (_REPO_ROOT / "intergrax/contracts/control_plane_mutation.py").is_file()
    assert (
        _REPO_ROOT
        / "intergrax/runtime/governance/control_plane_mutation_authorization.py"
    ).is_file()


def test_gr12_a1_scenario_cp_remains_gap_in_gov_final_catalog() -> None:
    from tests.qualification.governance.catalog import GOV_FINAL_4_SCENARIO_CATALOG

    cp = next(row for row in GOV_FINAL_4_SCENARIO_CATALOG if row.scenario_id == "CP")
    assert cp.result.name == "GAP"
    assert "GR-12" in (cp.notes or "")


def test_gr12_a1_gr10_remains_final_closed() -> None:
    assert "FINAL CLOSED" in GR10_OVERALL_FORMAL_CLOSURE.status
    assert GR10_POST_CLOSURE_NEXT_REMEDIATION.task_name.startswith("GR-12")


def test_gr12_a1_next_remediation_is_a2_composition() -> None:
    assert "GR-12-A2" in GR12_A1_NEXT_REMEDIATION.task_name
    assert "Composition" in GR12_A1_NEXT_REMEDIATION.task_name


def _module_path_for_entrypoint(entrypoint: str) -> Path:
    parts = entrypoint.split(".")
    assert parts[0] == "intergrax"
    for drop_suffix in (1, 2, 0):
        if drop_suffix == 0:
            candidate = parts
        else:
            candidate = parts[:-drop_suffix]
        py_path = _REPO_ROOT.joinpath(*candidate).with_suffix(".py")
        if py_path.is_file():
            return py_path
    return _REPO_ROOT.joinpath(*parts[:-1]).with_suffix(".py")


def test_gr12_a1_entrypoint_modules_exist_for_wired_surfaces() -> None:
    missing: list[str] = []
    for row in GR12_CONTROL_PLANE_SURFACES:
        if row.applicability is Gr12Applicability.NOT_APPLICABLE:
            continue
        py_path = _module_path_for_entrypoint(row.production_entrypoint)
        pkg_init = py_path.parent / "__init__.py"
        if not py_path.is_file() and not pkg_init.is_file():
            missing.append(str(py_path))
    assert missing == [], f"missing modules: {missing}"


def test_gr12_a1_inventory_covers_known_mutation_type_constants() -> None:
    mutations = {row.mutation for row in GR12_CONTROL_PLANE_SURFACES}
    for required in (
        "ahi.apply_profile",
        "ecp.scale_k8s_deployment",
        "task_control.cancel_task_execution",
        "agent_distribution.activate_runtime_revision",
    ):
        assert required in mutations


def test_gr12_a1_execution_exclusions_non_empty() -> None:
    assert GR12_EXECUTION_PLANE_EXCLUSIONS
    assert GR12_EXISTING_MECHANISMS
    assert GR12_CONSEQUENTIAL_MUTATION_DEFINITION.strip()
