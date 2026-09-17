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
from tests.qualification.governance.strategy.catalog import (
    GR10_FINAL_CAPABILITY_MATRIX,
    GR10_PRODUCTION_INVENTORY,
    Gr10CoverageStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_HOST_TASK = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "host_task.py"
_NEXUS_HOST = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "nexus_host_execution.py"
_HOST_WIRING = _REPO_ROOT / "intergrax" / "applications" / "_shared" / "host_task_execution_wiring.py"
_INFERENCE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "inference.py"

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
    assert "root_authority_admission=root_authority_admission" in source


def test_gr10_host_wiring_injects_root_admission_composition() -> None:
    source = _HOST_WIRING.read_text(encoding="utf-8-sig")
    assert "build_harness_root_execution_authority_admission" in source
    assert "RootExecutionAuthorityAdmissionPort" in source


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
