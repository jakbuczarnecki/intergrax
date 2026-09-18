# © Artur Czarnecki. All rights reserved.

"""GR-10-R6-R1 — mandatory Evidence composition + PRE_MODEL verdict semantics qualification."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.runtime.execution.inference_composition import build_governed_inference_executor
from intergrax.runtime.policy import pre_model_policy_evaluation as pre_model_module
from tests.qualification.governance.strategy.catalog import (
    GR10_INFERENCE_CAPABILITY_SEMANTICS,
    GR10_SCENARIO_CATALOG,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_matrix_inference_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSITION = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference_composition.py"
_COMPOSITION_TESTS = (
    _REPO_ROOT / "tests" / "unit" / "runtime" / "execution" / "test_inference_composition.py"
)
_INFERENCE_EXEC = (
    _REPO_ROOT / "tests" / "unit" / "runtime" / "execution" / "test_inference_executor.py"
)


def test_gr10_r6_r1_governed_builder_requires_evidence_persistence() -> None:
    param = inspect.signature(build_governed_inference_executor).parameters[
        "governance_evidence_persistence"
    ]
    assert param.default is inspect.Parameter.empty


def test_gr10_r6_r1_production_composition_proof_collectable() -> None:
    source = _COMPOSITION_TESTS.read_text(encoding="utf-8-sig")
    for name in (
        "test_build_governed_inference_executor_requires_persistence_port_signature",
        "test_governed_inference_executor_wires_custom_port",
        "test_test_only_helper_exists_outside_production_builder",
    ):
        assert f"def {name}" in source


def test_gr10_r6_r1_pre_model_verdict_proofs_collectable() -> None:
    source = _INFERENCE_EXEC.read_text(encoding="utf-8-sig")
    for name in (
        "test_inference_pre_model_require_human_emits_fact_before_fail_closed",
        "test_inference_pre_model_escalate_no_fact_fail_closed",
        "test_inference_pre_model_modify_no_fact_fail_closed",
    ):
        assert f"def {name}" in source


def test_gr10_r6_r1_pre_model_semantic_sets_explicit() -> None:
    assert hasattr(pre_model_module, "_PRE_MODEL_EVIDENCE_FACT_ACTIONS")
    assert hasattr(pre_model_module, "_PRE_MODEL_NO_EVIDENCE_FACT_ACTIONS")
    assert hasattr(pre_model_module, "_PRE_MODEL_EFFECTIVE_BLOCKED_ACTIONS")


def test_gr10_r6_r1_inf_e_catalog_includes_composition_and_verdict_proofs() -> None:
    entry = next(item for item in GR10_SCENARIO_CATALOG if item.scenario_id == "INF-E")
    joined = " ".join(entry.pytest_node_ids)
    assert "test_build_governed_inference_executor_requires_persistence_port_signature" in joined
    assert "test_inference_pre_model_require_human_emits_fact_before_fail_closed" in joined
    assert "test_inference_pre_model_escalate_no_fact_fail_closed" in joined
    assert "test_inference_pre_model_modify_no_fact_fail_closed" in joined


def test_gr10_r6_r1_governance_evidence_semantics_document_mandatory_composition() -> None:
    row = next(
        entry for entry in GR10_INFERENCE_CAPABILITY_SEMANTICS if entry.capability == "Governance Evidence"
    )
    assert row.applicability is Gr10Applicability.APPLICABLE
    assert row.coverage is Gr10CoverageStatus.QUALIFIED
    assert "mandatory" in row.reason.lower()
    assert gr10_matrix_inference_status("Governance Evidence") is Gr10CoverageStatus.QUALIFIED
