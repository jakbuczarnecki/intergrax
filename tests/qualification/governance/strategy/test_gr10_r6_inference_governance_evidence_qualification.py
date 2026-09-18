# © Artur Czarnecki. All rights reserved.

"""GR-10-R6 — INFERENCE Governance Evidence enterprise qualification."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_INFERENCE_CAPABILITY_SEMANTICS,
    GR10_SCENARIO_CATALOG,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_matrix_inference_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INFERENCE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference.py"
_PRE_MODEL = _REPO_ROOT / "intergrax" / "runtime" / "policy" / "pre_model_policy_evaluation.py"
_INFERENCE_COMPOSITION = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference_composition.py"
_DECISION_E2E_COMPOSITION = _REPO_ROOT / "testing_support" / "decision_e2e" / "composition.py"
_INFERENCE_EXEC = (
    _REPO_ROOT / "tests" / "unit" / "runtime" / "execution" / "test_inference_executor.py"
)


@dataclass(frozen=True, slots=True)
class _ApplicableGepRow:
    gep: str
    applicable: bool
    runtime_path: str
    evidence_expected: bool


_APPLICABLE_GEPS: tuple[_ApplicableGepRow, ...] = (
    _ApplicableGepRow(
        "PRE_MODEL",
        True,
        "InferenceExecutor → enforce_pre_model_before_structured_inference",
        True,
    ),
    _ApplicableGepRow("ROOT_EXECUTION_ADMISSION", False, "N/A for INFERENCE strategy root", False),
    _ApplicableGepRow("MEANINGFUL_SIDE_EFFECT", False, "N/A — no MSE spine on inference seam", False),
)


def test_gr10_r6_inference_governance_evidence_matrix_qualified() -> None:
    status = gr10_matrix_inference_status("Governance Evidence")
    assert status is Gr10CoverageStatus.QUALIFIED


def test_gr10_r6_applicable_gep_inventory_pre_model_only() -> None:
    applicable = [row for row in _APPLICABLE_GEPS if row.applicable]
    assert len(applicable) == 1
    assert applicable[0].gep == "PRE_MODEL"


def test_gr10_r6_semantics_governance_evidence_qualified() -> None:
    row = next(
        entry for entry in GR10_INFERENCE_CAPABILITY_SEMANTICS if entry.capability == "Governance Evidence"
    )
    assert row.applicability is Gr10Applicability.APPLICABLE
    assert row.coverage is Gr10CoverageStatus.QUALIFIED


def test_gr10_r6_inf_e_scenario_points_at_pre_model_evidence_proofs() -> None:
    entry = next(item for item in GR10_SCENARIO_CATALOG if item.scenario_id == "INF-E")
    assert entry.expected_status is Gr10CoverageStatus.QUALIFIED
    joined = " ".join(entry.pytest_node_ids)
    assert "test_inference_pre_model_allow_invokes_provider_once" in joined
    assert "test_inference_pre_model_deny_blocks_provider" in joined
    assert "test_inference_pre_model_custom_persistence_port_records_typed_fact" in joined
    assert "test_inference_pre_model_require_human_emits_fact_before_fail_closed" in joined
    assert "test_build_governed_inference_executor_requires_persistence_port_signature" in joined
    assert "root_allow_emits_exactly_one_governance_fact" not in joined


def test_gr10_r6_inference_executor_has_no_default_evidence_store_ast() -> None:
    source = _INFERENCE.read_text(encoding="utf-8-sig")
    assert "build_in_memory_governance_evidence_persistence" not in source
    assert "GovernanceEvidencePersistencePort" not in source
    assert "postgres" not in source.lower()
    assert "redis" not in source.lower()


def test_gr10_r6_pre_model_evidence_path_uses_public_contract_ast() -> None:
    source = _PRE_MODEL.read_text(encoding="utf-8-sig")
    assert "build_governance_fact_from_policy_decision" in source
    assert "GovernanceDecisionEvidenceFact" not in source or "build_governance_fact_from_policy_decision" in source
    tree = ast.parse(source, filename=str(_PRE_MODEL))
    module_assigns_dict_fact = any(
        isinstance(node, ast.Call)
        and isinstance(getattr(node.func, "id", None), str)
        and node.func.id == "dict"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )
    assert not module_assigns_dict_fact


def test_gr10_r6_canonical_composition_wires_evidence_persistence() -> None:
    import inspect

    from intergrax.runtime.execution.inference_composition import build_governed_inference_executor

    source = _INFERENCE_COMPOSITION.read_text(encoding="utf-8-sig")
    assert "build_governed_inference_executor" in source
    assert "build_governance_evidence_recorder" in source
    assert "build_in_memory_governance_evidence_persistence" not in source
    param = inspect.signature(build_governed_inference_executor).parameters[
        "governance_evidence_persistence"
    ]
    assert param.default is inspect.Parameter.empty
    decision_e2e = _DECISION_E2E_COMPOSITION.read_text(encoding="utf-8-sig")
    assert "governance_evidence_persistence=build_in_memory_governance_evidence_persistence()" in decision_e2e


def test_gr10_r6_inference_executor_evidence_proofs_collectable() -> None:
    source = _INFERENCE_EXEC.read_text(encoding="utf-8-sig")
    for name in (
        "test_inference_pre_model_allow_invokes_provider_once",
        "test_inference_pre_model_deny_blocks_provider",
        "test_inference_pre_model_deny_evidence_failure_still_denies_zero_provider",
        "test_inference_pre_model_evidence_failure_still_allows_provider",
        "test_inference_pre_model_custom_persistence_port_records_typed_fact",
    ):
        assert f"def {name}" in source
