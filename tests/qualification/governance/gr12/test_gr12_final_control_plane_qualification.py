# © Artur Czarnecki. All rights reserved.

"""GR-12-FINAL — parent control-plane governance qualification mechanical gate."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from tests.qualification.governance.catalog import (
    GOV_FINAL_4_SCENARIO_CATALOG,
    GovFinal4ScenarioResult,
)
from tests.qualification.governance.gr12.a3_path_qualifications import (
    GR12_A3_CORE_PATH_PROOFS,
)
from tests.qualification.governance.gr12.catalog import (
    GR12_CANONICAL_BOUNDARY_CLASS,
    GR12_CANONICAL_POLICY_PORT,
    GR12_CONTROL_PLANE_SURFACES,
    GR12_FINAL_BLOCKED_COVERAGE_STATUSES,
    GR12_FINAL_BYPASS_REGRESSION_PROOF_NODES,
    GR12_FINAL_EVIDENCE_PROOF_NODES,
    GR12_FINAL_HITL_NEGATIVE_PROOF_NODES,
    GR12_FINAL_PARENT_QUALIFICATION_STATUS,
    GR12_FINAL_PLUGINABILITY_PROOF_NODES,
    GR12_FINAL_QUALIFICATION_PROOF,
    GR12_FINAL_REPRESENTATIVE_EXECUTION_PROOF_NODES,
    GR12_FINAL_STALE_REVISION_NEGATIVE_PROOF_NODES,
    GR12_FINAL_TENANT_SCOPE_NEGATIVE_PROOF_NODES,
    GR12_GOV_FINAL_4_CP_QUALIFICATION_CANDIDATE_STATUS,
    Gr12Applicability,
    Gr12CoverageStatus,
    gr12_applicable_surfaces,
    gr12_path_ids,
)
from tests.qualification.governance.gr12.qualification_support import (
    assert_gr12_a3_path_semantic_integrity,
    assert_proof_nodes_registered,
)
from tests.unit.applications.test_catalog_hot_reload_bypass_inventory import (
    _production_reload_bypass_offenders,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_GOVERNANCE_ROOT = _REPO_ROOT / "intergrax" / "runtime" / "governance"
_CLA04_CONTRACT = _REPO_ROOT / "intergrax" / "contracts" / "control_plane_mutation.py"
_FORBIDDEN_PARENT_STATUSES = frozenset({"CLOSED", "FINAL CLOSED"})


def _resolve_qualification_proof(proof: str) -> None:
    proof = proof.strip()
    if not proof:
        return
    if "::" in proof:
        assert_proof_nodes_registered((proof,))
        return
    path = _REPO_ROOT / proof
    assert path.is_file(), proof


def _module_path_for_entrypoint(entrypoint: str) -> Path:
    parts = entrypoint.split(".")
    assert parts[0] == "intergrax"
    for drop_suffix in (1, 2, 0):
        candidate = parts[:-drop_suffix] if drop_suffix else parts
        py_path = _REPO_ROOT.joinpath(*candidate).with_suffix(".py")
        if py_path.is_file():
            return py_path
    return _REPO_ROOT.joinpath(*parts[:-1]).with_suffix(".py")


def test_gr12_final_f01_path_ids_unique() -> None:
    ids = gr12_path_ids()
    assert len(ids) == len(set(ids))


def test_gr12_final_f02_applicable_rows_are_qualified() -> None:
    for row in gr12_applicable_surfaces():
        assert row.coverage is Gr12CoverageStatus.QUALIFIED, row.path_id


def test_gr12_final_f03_applicable_rows_have_qualification_proof() -> None:
    for row in gr12_applicable_surfaces():
        assert row.qualification_proof.strip(), row.path_id


def test_gr12_final_f04_qualification_proof_artifacts_exist() -> None:
    for row in GR12_CONTROL_PLANE_SURFACES:
        if row.coverage is Gr12CoverageStatus.QUALIFIED:
            _resolve_qualification_proof(row.qualification_proof)


def test_gr12_final_f05_not_applicable_rows_have_explicit_remediation() -> None:
    for row in GR12_CONTROL_PLANE_SURFACES:
        if row.applicability is not Gr12Applicability.NOT_APPLICABLE:
            continue
        assert row.coverage is Gr12CoverageStatus.NOT_APPLICABLE, row.path_id
        assert row.future_remediation.strip(), row.path_id
        assert row.recommended_owner.strip(), row.path_id


def test_gr12_final_f06_through_f09_no_blocked_coverage_status() -> None:
    offenders: list[str] = []
    for row in GR12_CONTROL_PLANE_SURFACES:
        if row.coverage in GR12_FINAL_BLOCKED_COVERAGE_STATUSES:
            offenders.append(f"{row.path_id}:{row.coverage}")
    assert offenders == []


def test_gr12_final_f10_canonical_cla04_contract_family_single() -> None:
    assert _CLA04_CONTRACT.is_file()
    source = _CLA04_CONTRACT.read_text(encoding="utf-8")
    for symbol in (
        "ControlPlaneMutationRequest",
        "ControlPlaneMutationAuthorizationEvidence",
        "ControlPlaneMutationPolicyEvaluator",
        "ControlPlaneMutationAuthorizationResult",
    ):
        assert f"class {symbol}" in source
    assert "class ControlPlaneMutationPolicyEvaluator(Protocol):" in source
    assert "ControlPlaneMutationAuthorizationBoundary" in GR12_CANONICAL_BOUNDARY_CLASS
    assert "ControlPlaneMutationPolicyEvaluator" in GR12_CANONICAL_POLICY_PORT
    boundary_module = (
        _REPO_ROOT
        / "intergrax/runtime/governance/control_plane_mutation_authorization.py"
    )
    assert boundary_module.is_file()
    assert (
        "class ControlPlaneMutationAuthorizationBoundary"
        in boundary_module.read_text(encoding="utf-8")
    )


def test_gr12_final_f11_no_universal_governance_engine() -> None:
    for path in _GOVERNANCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "GovernanceEngine":
                raise AssertionError(f"forbidden GovernanceEngine in {path}")


def test_gr12_final_f12_no_universal_control_plane_mutation_executor() -> None:
    forbidden = re.compile(
        r"class\s+(Universal|Global)(ControlPlane)?MutationExecutor\b",
        re.MULTILINE,
    )
    for root in (_REPO_ROOT / "intergrax",):
        for path in root.rglob("*.py"):
            if "/tests/" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8")
            assert not forbidden.search(text), path


def test_gr12_final_f13_applicable_surfaces_name_domain_mutation_owner() -> None:
    for row in gr12_applicable_surfaces():
        assert row.recommended_owner.strip(), row.path_id
        assert row.mutation.strip(), row.path_id


def test_gr12_final_f14_catalog_hot_reload_qualified() -> None:
    row = next(
        r
        for r in GR12_CONTROL_PLANE_SURFACES
        if r.path_id == "CP-PLUGIN-CATALOG-HOT-RELOAD"
    )
    assert row.coverage is Gr12CoverageStatus.QUALIFIED


def test_gr12_final_f15_vector_admin_qualified() -> None:
    row = next(
        r for r in GR12_CONTROL_PLANE_SURFACES if r.path_id == "CP-VECTOR-INDEX-ADMIN"
    )
    assert row.coverage is Gr12CoverageStatus.QUALIFIED


def test_gr12_final_f16_memory_path_not_applicable() -> None:
    row = next(
        r
        for r in GR12_CONTROL_PLANE_SURFACES
        if r.path_id == "CP-MEM-SPECIALIZED-MUTATION"
    )
    assert row.applicability is Gr12Applicability.NOT_APPLICABLE
    assert row.coverage is Gr12CoverageStatus.NOT_APPLICABLE


def test_gr12_final_f17_no_live_operator_memory_surface_regression() -> None:
    from tests.qualification.governance.gr12.test_gr12_a4_r3_r1_memory_specialized_governance_qualification import (
        test_gr12_a4_r3_r1_m5_no_live_operator_memory_mutation_surface,
    )

    test_gr12_a4_r3_r1_m5_no_live_operator_memory_mutation_surface()


def test_gr12_final_f18_composition_omission_fail_closed_proofs() -> None:
    for path_id in ("CP-HOST-BOUNDARY-OPTIONAL", "CP-ECP-BOUNDARY-OPTIONAL"):
        bundle = next(b for b in GR12_A3_CORE_PATH_PROOFS if b.path_id == path_id)
        assert_gr12_a3_path_semantic_integrity(bundle)


def test_gr12_final_f19_custom_evaluator_replaceability_proofs_registered() -> None:
    assert_proof_nodes_registered(GR12_FINAL_PLUGINABILITY_PROOF_NODES)


def test_gr12_final_f20_tenant_scope_negative_proofs_registered() -> None:
    assert_proof_nodes_registered(GR12_FINAL_TENANT_SCOPE_NEGATIVE_PROOF_NODES)


def test_gr12_final_f21_stale_revision_negative_proofs_registered() -> None:
    assert_proof_nodes_registered(GR12_FINAL_STALE_REVISION_NEGATIVE_PROOF_NODES)


def test_gr12_final_f22_hitl_does_not_equate_approval_with_permission() -> None:
    assert_proof_nodes_registered(GR12_FINAL_HITL_NEGATIVE_PROOF_NODES)


def test_gr12_final_f23_control_plane_evidence_proofs_registered() -> None:
    assert_proof_nodes_registered(GR12_FINAL_EVIDENCE_PROOF_NODES)


def test_gr12_final_f24_catalogued_surfaces_match_bypass_inventory() -> None:
    assert_proof_nodes_registered(GR12_FINAL_BYPASS_REGRESSION_PROOF_NODES)
    assert _production_reload_bypass_offenders() == []
    missing_modules: list[str] = []
    for row in gr12_applicable_surfaces():
        py_path = _module_path_for_entrypoint(row.production_entrypoint)
        if not py_path.is_file() and not (py_path.parent / "__init__.py").is_file():
            missing_modules.append(row.path_id)
    assert missing_modules == []


def test_gr12_final_f25_parent_status_closed_with_independent_audit_sha() -> None:
    from tests.qualification.governance.gr12.catalog import (
        GR12_INDEPENDENT_AUDIT_ACCEPTANCE_SHA,
        GR12_SEMANTIC_QUALIFICATION_BASELINE_SHA,
    )

    assert GR12_FINAL_PARENT_QUALIFICATION_STATUS == "CLOSED"
    assert GR12_SEMANTIC_QUALIFICATION_BASELINE_SHA == (
        "b706c2c72a900575ec360b7f217c97a3656c71b9"
    )
    assert GR12_INDEPENDENT_AUDIT_ACCEPTANCE_SHA == (
        "03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06"
    )
    assert (_REPO_ROOT / GR12_FINAL_QUALIFICATION_PROOF).is_file()


def test_gr12_final_gov_final_4_cp_qualified_after_independent_audit() -> None:
    assert GR12_GOV_FINAL_4_CP_QUALIFICATION_CANDIDATE_STATUS == "QUALIFIED"
    cp = next(row for row in GOV_FINAL_4_SCENARIO_CATALOG if row.scenario_id == "CP")
    assert cp.result is GovFinal4ScenarioResult.QUALIFIED
    assert cp.primary_pytest_node_ids


def test_gr12_final_representative_execution_proof_nodes_collect() -> None:
    assert_proof_nodes_registered(GR12_FINAL_REPRESENTATIVE_EXECUTION_PROOF_NODES)
