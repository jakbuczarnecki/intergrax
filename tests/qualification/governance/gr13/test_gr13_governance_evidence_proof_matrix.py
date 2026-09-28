# © Artur Czarnecki. All rights reserved.

"""GR-13 mechanical proof matrix gates (G13-01..G13-14)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.qualification.governance.catalog import (
    GOV_FINAL_4_SCENARIO_CATALOG,
    GovFinal4ScenarioResult,
)
from tests.qualification.governance.gr13.catalog import (
    GR11_ACCEPTED_HEAD,
    GR12_ACCEPTED_HEAD,
    GR13_BASELINE_HEAD,
    GR13_FACT_CONTRACT,
    GR13_GR8_EVIDENCE_FAILURE_PROOF_NODES,
    GR13_HELPER_ONLY_FORBIDDEN_MARKERS,
    GR13_PERSISTENCE_CONTRACT,
    GR13_PROOF_MATRIX,
    GR13_REGISTERED_EMISSION_MODULES,
    GR13_SCENARIO_Y_PROOF_NODES,
    Gr13ProofPathKind,
    gr13_all_proof_nodes,
    gr13_positive_proof_test_name,
)
from tests.qualification.governance.strategy.catalog import (
    GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED,
    GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED,
    Gr10EvidenceCertificationRequirement,
    gr10_agentic_gep_semantics,
    gr10_orchestration_gep_semantics,
)
from tests.qualification.governance.pytest_node_integrity import (
    pytest_nodes_missing_from_collection,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GR13_UNIT = _REPO_ROOT / (
    "tests/unit/runtime/governance/test_gr13_governance_evidence_gep_emission.py"
)


def _proof_source_for_row(module_source: str, test_name: str) -> str:
    body = _extract_test_function_source(module_source, test_name)
    for helper in (
        "_prove_tool_plan_or_access_emits_fact",
        "_prove_tool_invocation_policy_emits_fact",
    ):
        if helper in body:
            body = f"{body}\n{_extract_test_function_source(module_source, helper)}"
    return body


def _extract_test_function_source(module_source: str, test_name: str) -> str:
    lines = module_source.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        if f"def {test_name}(" in line:
            start = index
            break
    assert start is not None, f"missing test function {test_name}"
    collected: list[str] = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith(("def ", "async def ", "@pytest")):
            break
        collected.append(line)
    return "\n".join(collected)


def test_g13_01_closed_world_inventory_matches_gr10_deferred() -> None:
    agentic = {row.gep for row in GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED}
    orch = {row.gep for row in GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED}
    matrix_agentic = {row.gep for row in GR13_PROOF_MATRIX if row.strategy == "AGENTIC"}
    matrix_orch = {
        row.gep for row in GR13_PROOF_MATRIX if row.strategy == "ORCHESTRATION"
    }
    assert matrix_agentic == agentic
    assert matrix_orch == orch
    assert len(agentic) == 7
    assert len(orch) == 5


def test_g13_02_all_required_rows_mechanically_proven() -> None:
    unit_source = _GR13_UNIT.read_text(encoding="utf-8")
    for row in GR13_PROOF_MATRIX:
        sem = (
            gr10_agentic_gep_semantics(row.gep)
            if row.strategy == "AGENTIC"
            else gr10_orchestration_gep_semantics(row.gep)
        )
        assert (
            sem.gr13_evidence_requirement
            is Gr10EvidenceCertificationRequirement.REQUIRED_IN_GR13
        )
        test_name = gr13_positive_proof_test_name(row.strategy, row.gep)
        body = _proof_source_for_row(unit_source, test_name)
        for marker in row.canonical_path_markers:
            assert marker in body, f"{row.strategy}/{row.gep} missing marker {marker}"
        for forbidden in GR13_HELPER_ONLY_FORBIDDEN_MARKERS:
            if forbidden in body:
                assert any(m in body for m in row.canonical_path_markers)


def test_g13_03_canonical_fact_contract_on_all_rows() -> None:
    for row in GR13_PROOF_MATRIX:
        assert row.fact_contract == GR13_FACT_CONTRACT


def test_g13_04_canonical_persistence_contract_on_all_rows() -> None:
    for row in GR13_PROOF_MATRIX:
        assert row.persistence_contract == GR13_PERSISTENCE_CONTRACT


def test_g13_05_exact_canonical_owner_and_emission_module() -> None:
    for row in GR13_PROOF_MATRIX:
        sem = (
            gr10_agentic_gep_semantics(row.gep)
            if row.strategy == "AGENTIC"
            else gr10_orchestration_gep_semantics(row.gep)
        )
        assert row.canonical_owner == sem.canonical_owner
        assert row.production_path == sem.production_path
        assert row.emission_module in GR13_REGISTERED_EMISSION_MODULES


def test_g13_06_shared_strategy_rows_prove_same_canonical_module() -> None:
    by_gep: dict[str, list[str]] = {}
    for row in GR13_PROOF_MATRIX:
        if row.proof_path_kind is not Gr13ProofPathKind.SHARED_CANONICAL_PATH:
            continue
        by_gep.setdefault(row.gep, []).append(row.emission_module)
    for gep, modules in by_gep.items():
        assert len(modules) == 2
        assert modules[0] == modules[1], gep
        markers = {
            row.strategy: row.canonical_path_markers
            for row in GR13_PROOF_MATRIX
            if row.gep == gep
        }
        assert len(markers) == 2
        assert markers["AGENTIC"] == markers["ORCHESTRATION"]


def test_g13_07_negative_decision_coverage_declared() -> None:
    for row in GR13_PROOF_MATRIX:
        assert row.negative_proof_nodes


def test_g13_08_evidence_non_authoritative_proof_nodes_registered() -> None:
    nodes = gr13_all_proof_nodes()
    assert (
        "tests/unit/runtime/governance/test_gr13_governance_evidence_gep_emission.py::"
        "test_gr13_evidence_does_not_grant_permission" in nodes
    )


def test_g13_09_identity_correlation_positive_proofs_present() -> None:
    unit_source = _GR13_UNIT.read_text(encoding="utf-8")
    assert "assert_gr13_fact_identity" in unit_source


def test_g13_10_no_duplicate_evidence_semantics_in_registered_modules() -> None:
    for rel in sorted(GR13_REGISTERED_EMISSION_MODULES):
        path = _REPO_ROOT / rel
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in {
                "GovernanceDecisionEvidenceFact",
                "GovernanceEvidencePersistencePort",
            }:
                pytest.fail(f"duplicate evidence contract class in {rel}")


def test_g13_11_gr8_evidence_failure_nodes_collectable() -> None:
    missing, proc = pytest_nodes_missing_from_collection(
        GR13_GR8_EVIDENCE_FAILURE_PROOF_NODES,
        _REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, missing


def test_g13_12_proof_nodes_collectable() -> None:
    missing, proc = pytest_nodes_missing_from_collection(
        gr13_all_proof_nodes(),
        _REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not missing, missing


def test_g13_13_gr11_gr12_status_reconciled() -> None:
    assert GR11_ACCEPTED_HEAD.startswith("4e5878f")
    assert GR12_ACCEPTED_HEAD.startswith("03dde6c6")


def test_g13_14_scenario_y_bound_to_recovery_proofs() -> None:
    scenario_y = next(
        row for row in GOV_FINAL_4_SCENARIO_CATALOG if row.scenario_id == "Y"
    )
    assert scenario_y.result is GovFinal4ScenarioResult.QUALIFIED
    assert tuple(scenario_y.primary_pytest_node_ids) == GR13_SCENARIO_Y_PROOF_NODES


def test_g13_baseline_head_documented() -> None:
    assert GR13_BASELINE_HEAD.startswith("4e5878f")
