# © Artur Czarnecki. All rights reserved.

"""GR-10 architecture gates — production strategy entry and composition truth."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.root_execution_operation import RootExecutionOperation
from intergrax.runtime.execution.root_execution_operation_mapping import (
    root_execution_operation_from_request,
)
from intergrax.runtime.execution.request import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.strategy import ExecutionStrategy, execution_strategy_from_capabilities
from tests.qualification.governance.strategy.gr10_inference_current_doc_ssot import (
    gr10_assert_current_docs_inference_ssot,
    gr10_architecture_remaining_gaps_slice,
    gr10_forbidden_inference_gap_hits,
    gr10_maintainer_roadmap_slice,
)
from tests.qualification.governance.strategy.catalog import (
    GR10_AGENTIC_CAPABILITY_SEMANTICS,
    GR10_FINAL_CAPABILITY_MATRIX,
    GR10_INFERENCE_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_PRODUCTION_INVENTORY,
    GR10_SCENARIO_CATALOG,
    Gr10Applicability,
    Gr10CoverageStatus,
    gr10_matrix_agentic_status,
    gr10_matrix_inference_status,
    gr10_matrix_orchestration_status,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST_TASK = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_NEXUS_HOST = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "nexus_host_execution.py"
_HOST_WIRING = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "host_task_execution_wiring.py"
_INFERENCE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference.py"
_INFERENCE_EXEC = (
    _REPO_ROOT / "tests" / "unit" / "runtime" / "execution" / "test_inference_executor.py"
)

_FORBIDDEN_SERVICE_LOCATOR_NAMES = frozenset(
    {
        "get_governance",
        "get_policy_engine",
        "global_current_governance",
    },
)


def test_gr10_production_inventory_covers_three_strategies() -> None:
    strategies = {entry.strategy for entry in GR10_PRODUCTION_INVENTORY}
    assert strategies == {"INFERENCE", "AGENTIC", "ORCHESTRATION"}


def test_gr10_capability_matrix_covers_required_rows() -> None:
    capabilities = {row.capability for row in GR10_FINAL_CAPABILITY_MATRIX}
    required = {
        "Root admission",
        "Inner Governance",
        "Policy evaluation",
        "MSE",
        "Decision-bound effect",
        "HITL",
        "Continuation",
        "Reliability",
        "Governance Evidence",
    }
    assert required <= capabilities


def test_gr10_matrix_not_all_qualified_honesty() -> None:
    """GR-10 must not overclaim enterprise closure."""
    cells = GR10_FINAL_CAPABILITY_MATRIX
    qualified_count = sum(
        1
        for row in cells
        for status in (row.inference, row.agentic, row.orchestration)
        if status is Gr10CoverageStatus.QUALIFIED
    )
    gap_or_partial = sum(
        1
        for row in cells
        for status in (row.inference, row.agentic, row.orchestration)
        if status
        in (
            Gr10CoverageStatus.GAP,
            Gr10CoverageStatus.PARTIAL,
            Gr10CoverageStatus.WIRED_NOT_QUALIFIED,
        )
    )
    assert gap_or_partial > 0
    assert qualified_count > 0


def test_gr10_host_task_wires_mandatory_root_launcher() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8-sig")
    assert "DefaultRootExecutionLauncher" in source
    assert "RootExecutionLaunchRequest" in source
    assert "root_execution_operation_from_request" in source
    assert "RootExecutionAuthorityAdmissionPort" in source


def test_gr10_host_task_capabilities_agent_or_orchestration_only() -> None:
    source = _HOST_TASK.read_text(encoding="utf-8-sig")
    assert "ExecutionCapability.ORCHESTRATION" in source
    assert "ExecutionCapability.AGENT" in source
    assert "inference_executor" not in source


def test_gr10_nexus_host_execution_requires_root_admission_parameter() -> None:
    source = _NEXUS_HOST.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_NEXUS_HOST))
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "build_host_task_execution"
    )
    kwonly = [arg.arg for arg in fn.args.kwonlyargs]
    assert "root_authority_admission" in kwonly
    assert "admit_root_governance_identity" in kwonly
    assert "root_authority_admission=root_authority_admission" in source
    assert "admit_certified_internal_harness_root_governance_identity" not in source


def test_gr10_host_wiring_requires_explicit_root_admission_parameters() -> None:
    source = _HOST_WIRING.read_text(encoding="utf-8-sig")
    assert "RootExecutionAuthorityAdmissionPort" in source
    assert "admit_root_governance_identity" in source
    assert "admit_harness_root_governance_identity" not in source


def test_gr10_strategy_operation_mapping_for_all_strategies() -> None:
    inference_req = ExecutionRequest(input=(), output_type=dict)
    agent_req = ExecutionRequest(
        input=(),
        output_type=dict,
        capabilities=frozenset({ExecutionCapability.AGENT}),
    )
    orch_req = ExecutionRequest(
        input=(),
        output_type=dict,
        capabilities=frozenset({ExecutionCapability.ORCHESTRATION}),
    )
    assert execution_strategy_from_capabilities(inference_req.capabilities) is ExecutionStrategy.INFERENCE
    assert execution_strategy_from_capabilities(agent_req.capabilities) is ExecutionStrategy.AGENTIC
    assert (
        execution_strategy_from_capabilities(orch_req.capabilities)
        is ExecutionStrategy.ORCHESTRATION
    )
    assert (
        root_execution_operation_from_request(inference_req)
        is RootExecutionOperation.ROOT_INFERENCE
    )
    assert root_execution_operation_from_request(agent_req) is RootExecutionOperation.ROOT_AGENT
    assert (
        root_execution_operation_from_request(orch_req)
        is RootExecutionOperation.ROOT_ORCHESTRATION
    )


def test_gr10_inference_executor_requires_active_identity_no_direct_bypass_tokens() -> None:
    source = _INFERENCE.read_text(encoding="utf-8-sig")
    assert "require_active_execution_identity" in source
    assert "require_active_execution_id" in source
    for forbidden in _FORBIDDEN_SERVICE_LOCATOR_NAMES:
        assert forbidden not in source


def test_gr10_host_execution_wiring_no_service_locator() -> None:
    for path in (_HOST_TASK, _HOST_WIRING, _NEXUS_HOST):
        source = path.read_text(encoding="utf-8-sig")
        for forbidden in _FORBIDDEN_SERVICE_LOCATOR_NAMES:
            assert forbidden not in source


def test_gr10_agentic_inventory_and_matrix_semantics_are_consistent() -> None:
    matrix_by_cap = {row.capability: row.agentic for row in GR10_FINAL_CAPABILITY_MATRIX}
    assert {row.capability for row in GR10_AGENTIC_CAPABILITY_SEMANTICS} == matrix_by_cap.keys()
    for row in GR10_AGENTIC_CAPABILITY_SEMANTICS:
        assert matrix_by_cap[row.capability] is gr10_matrix_agentic_status(row.capability)


def test_gr10_orchestration_inventory_and_matrix_semantics_are_consistent() -> None:
    matrix_by_cap = {row.capability: row.orchestration for row in GR10_FINAL_CAPABILITY_MATRIX}
    assert {row.capability for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS} == matrix_by_cap.keys()
    for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS:
        assert matrix_by_cap[row.capability] is gr10_matrix_orchestration_status(row.capability)


def test_gr10_inference_inventory_and_matrix_semantics_are_consistent() -> None:
    semantics = {row.capability: row for row in GR10_INFERENCE_CAPABILITY_SEMANTICS}
    matrix_by_cap = {row.capability: row.inference for row in GR10_FINAL_CAPABILITY_MATRIX}
    assert semantics.keys() == matrix_by_cap.keys()
    for capability, expected in semantics.items():
        assert matrix_by_cap[capability] is gr10_matrix_inference_status(capability)
        if expected.applicability is Gr10Applicability.NOT_APPLICABLE:
            assert matrix_by_cap[capability] is Gr10CoverageStatus.NOT_APPLICABLE
            assert expected.coverage is None
            assert matrix_by_cap[capability] is not Gr10CoverageStatus.GAP
        else:
            assert expected.coverage is matrix_by_cap[capability]


def test_gr10_inventory_and_capability_matrix_are_consistent() -> None:
    """NOT_APPLICABLE in production inventory must not pair with GAP in the matrix."""
    inference_inventory = next(
        entry for entry in GR10_PRODUCTION_INVENTORY if entry.strategy == "INFERENCE"
    )
    for label, path in (
        ("hitl", inference_inventory.hitl_path),
        ("reliability", inference_inventory.reliability_path),
    ):
        assert "NOT_APPLICABLE" in path, label
    matrix = {row.capability: row.inference for row in GR10_FINAL_CAPABILITY_MATRIX}
    assert matrix["HITL"] is Gr10CoverageStatus.NOT_APPLICABLE
    assert matrix["Reliability"] is Gr10CoverageStatus.NOT_APPLICABLE
    for row in GR10_FINAL_CAPABILITY_MATRIX:
        if row.inference is Gr10CoverageStatus.NOT_APPLICABLE:
            assert row.inference is not Gr10CoverageStatus.GAP


def test_gr10_not_applicable_capabilities_are_not_enterprise_blockers() -> None:
    for row in GR10_INFERENCE_CAPABILITY_SEMANTICS:
        if row.applicability is Gr10Applicability.NOT_APPLICABLE:
            assert row.coverage is None


def test_gr10_inference_executor_enforces_pre_model_before_provider_ast() -> None:
    source = _INFERENCE.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_INFERENCE))
    executor_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "InferenceExecutor"
    )
    execute_fn = next(
        node
        for node in executor_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute"
    )
    policy_index = None
    invoke_index = None
    for index, node in enumerate(execute_fn.body):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "enforce_pre_model_before_structured_inference":
                policy_index = index
        if isinstance(node, ast.FunctionDef) and node.name == "_invoke":
            invoke_index = index
    assert policy_index is not None
    assert invoke_index is not None
    assert policy_index < invoke_index
    assert "RuntimePolicyEngine" not in source
    assert "PolicyEngine()" not in source


def test_gr10_inf_premodel_scenarios_reference_inference_executor() -> None:
    allow = next(
        entry for entry in GR10_SCENARIO_CATALOG if entry.scenario_id == "INF-PREMODEL-ALLOW"
    )
    deny = next(
        entry for entry in GR10_SCENARIO_CATALOG if entry.scenario_id == "INF-PREMODEL-DENY"
    )
    assert allow.expected_status is Gr10CoverageStatus.QUALIFIED
    assert deny.expected_status is Gr10CoverageStatus.QUALIFIED
    assert _INFERENCE_EXEC.name in allow.pytest_node_ids[0]


_QUAL_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "GOVERNANCE_FINAL_E2E_QUALIFICATION.md"
)


def test_gr10_current_doc_status_gr8_closed_gr10_partial() -> None:
    text = _QUAL_DOC.read_text(encoding="utf-8-sig")
    section_start = text.find("## Current qualification status")
    assert section_start != -1
    section = text[section_start : section_start + 1200]
    assert "GR-8" in section and "CLOSED" in section
    assert "GR-10" in section and "PARTIAL" in section
    assert "GR-8 | OPEN" not in section


def test_gr10_r2_adr1_r1_current_docs_inference_ssot_semantic_gate() -> None:
    """GR10_INFERENCE_CAPABILITY_SEMANTICS → current-status governance docs."""
    gr10_assert_current_docs_inference_ssot()


def test_gr10_r2_adr1_r1_current_docs_do_not_list_inference_hitl_as_gap() -> None:
    for label, body in (
        ("roadmap", gr10_maintainer_roadmap_slice()),
        ("arch_gaps", gr10_architecture_remaining_gaps_slice()),
    ):
        assert "INFERENCE HITL/Reliability" not in body, label
        assert "INFERENCE HITL residual" not in body.lower(), label
        assert not gr10_forbidden_inference_gap_hits(body), label


def test_gr10_r2_adr1_r1_current_docs_do_not_list_inference_reliability_as_gap() -> None:
    roadmap = gr10_maintainer_roadmap_slice()
    assert "INFERENCE Reliability" not in roadmap
    assert "HITL/Reliability residual" not in roadmap


def test_gr10_final_current_docs_record_inference_pre_model_qualified() -> None:
    arch = gr10_architecture_remaining_gaps_slice()
    plan = gr10_maintainer_roadmap_slice()
    assert "PRE_MODEL" in arch and "QUALIFIED" in arch
    assert "GR-10-FINAL" in arch or "GR-10-R2" in arch
    assert "GR-10-FINAL" in plan or "GR-10-R3" in plan
