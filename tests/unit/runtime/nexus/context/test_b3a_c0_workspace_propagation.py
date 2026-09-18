# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3A-C0 — canonical workspace_id propagation to ContextAssemblyRequest."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.context_assembly_bridge import build_acp_assembly_request
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.nexus.context.graph_assembly import build_graph_assembly_request
from intergrax.runtime.nexus.context.uaep_assemble import build_uaep_assembly_request
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.task.task import Task

pytestmark = pytest.mark.unit


def test_c0_canonical_propagation_runtime_request_to_assembly() -> None:
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        tenant_id="tenant-A",
        workspace_id="workspace-A",
    )
    assembly = build_uaep_assembly_request(request, agent_id="agent-1")
    assert assembly.workspace_id == "workspace-A"
    assert assembly.tenant_id == "tenant-A"


def test_c0_workspace_independent_of_context_scope_binding() -> None:
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-A",
        workspace_id="ws-A",
        assembly_scope="graph_node",
        objective="work",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert request.workspace_id == "ws-A"
    assert request.trace_id == "trace-1"


def test_c0_missing_workspace_non_workspace_flow_valid() -> None:
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-A",
        assembly_scope="acp_step",
        objective="work",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert request.workspace_id is None


def test_c0_artifact_path_can_detect_missing_workspace() -> None:
    request = ContextAssemblyRequest(
        trace_id="trace-1",
        run_id="run-1",
        task_id="task-1",
        tenant_id="tenant-A",
        assembly_scope="graph_node",
        objective="work",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert request.workspace_id is None or not request.workspace_id.strip()


def test_c0_graph_assembly_uses_task_envelope_not_metadata() -> None:
    task = Task.from_envelope(
        TaskEnvelope(
            tenant_id="tenant-A",
            user_id="user-1",
            message="do",
            workspace_id="workspace-9",
            metadata={"workspace_id": "metadata-spoof"},
        )
    )
    node = ExecutionNode(node_id="n1", capability="llm")
    from intergrax.runtime.nexus.context.context_budget import ContextBudgetPolicy

    run_id = mint_run_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=mint_attempt_id())
    try:
        assembly = build_graph_assembly_request(
            task,
            node,
            policy=TaskContextAssemblyOptions(),
            budget_policy=ContextBudgetPolicy(max_chars=4000),
        )
    finally:
        reset_active_execution_identity(token)
    assert assembly.workspace_id == "workspace-9"


def test_c0_acp_bridge_propagates_step_workspace() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        tenant_id="tenant-A",
        workspace_id="workspace-A",
        task_id="task_abc123",
        run_id=mint_run_id(),
    )
    assembly = build_acp_assembly_request(step_ctx)
    assert assembly.workspace_id == "workspace-A"


def test_c0_rejects_blank_workspace_id() -> None:
    with pytest.raises(ValueError, match="workspace_id"):
        ContextAssemblyRequest(
            trace_id="trace-1",
            run_id="run-1",
            task_id="task-1",
            tenant_id="tenant-A",
            workspace_id="   ",
            assembly_scope="acp_step",
            objective="work",
            decision_profile=ContextDecisionSnapshot(),
            budget_policy=ContextBudgetSnapshot(),
            assembly_options=TaskContextAssemblyOptions(),
        )
