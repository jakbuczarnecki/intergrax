# © Artur Czarnecki. All rights reserved.

"""MP-5F-B3A-C2 — complete typed workspace authority propagation gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from intergrax.runtime.nexus.agents.acp_uaep_shim import build_step_context_from_uaep
from intergrax.agents.authoring.context_assembly_bridge import build_acp_assembly_request
from intergrax.runtime.nexus.agents.uaep_step_bridge import build_uaep_step_context
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_step import AgentStep
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
)
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.runtime.context_lifecycle.contracts import UclArtifactOwnershipScope
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from intergrax.runtime.nexus.context.graph_assembly import build_graph_assembly_request
from intergrax.runtime.nexus.context.uaep_assemble import build_uaep_assembly_request
from intergrax.runtime.nexus.context.ucl_artifact_ownership_composition import (
    resolve_ucl_artifact_ownership_scope,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    reset_active_execution_identity,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_PRODUCTION_BUILDER_PATHS = (
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "graph_assembly.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "uaep_assemble.py",
    _REPO_ROOT / "intergrax" / "runtime" / "nexus" / "context" / "iterative_tool_context_assembly.py",
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "context_assembly_bridge.py",
)
_AUTHORITY_PROPAGATION_PATHS = _PRODUCTION_BUILDER_PATHS + (
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "uaep_step_bridge.py",
    _REPO_ROOT / "intergrax" / "agents" / "authoring" / "acp_uaep_shim.py",
)


def test_c2_runtime_request_envelope_roundtrip_preserves_workspace() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    envelope = TaskEnvelope(
        tenant_id="tenant-A",
        user_id="user-1",
        message="work",
        workspace_id="ws-roundtrip",
    )
    request = RuntimeRequest.from_envelope(envelope, task_id=task_id, run_id=run_id)
    assert request.workspace_id == "ws-roundtrip"
    assert request.to_envelope().workspace_id == "ws-roundtrip"


def test_c2_uaep_assembly_preserves_runtime_request_workspace() -> None:
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        tenant_id="tenant-A",
        workspace_id="ws-A",
    )
    assembly = build_uaep_assembly_request(request, agent_id="agent-1")
    assert assembly.workspace_id == "ws-A"


def test_c2_graph_assembly_preserves_task_envelope_workspace() -> None:
    task = Task.from_envelope(
        TaskEnvelope(
            tenant_id="tenant-A",
            user_id="user-1",
            message="do",
            workspace_id="ws-graph",
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
    assert assembly.workspace_id == "ws-graph"


def test_c2_acp_bridge_preserves_step_workspace() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        tenant_id="tenant-A",
        workspace_id="ws-A",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    assembly = build_acp_assembly_request(step_ctx)
    assert assembly.workspace_id == "ws-A"


def test_c2_uaep_step_bridge_preserves_exec_ctx_workspace() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="tenant-A",
        workspace_id="ws-uaep",
    )
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="agent-1",
        workspace_id=request.workspace_id,
        request=request,
    )
    kernel_ctx = StepKernelContext(
        agent_id="agent-1",
        run_id=run_id,
        task_id=task_id,
        tenant_id="tenant-A",
        max_steps=8,
        policy_engine=None,
        state_root={},
    )
    step = AgentStep(step_index=0, step_id="s0", step_name="llm")
    step_ctx = build_uaep_step_context(step, exec_ctx, kernel_ctx)
    assembly = build_acp_assembly_request(step_ctx)
    assert assembly.workspace_id == "ws-uaep"


def test_c2_acp_uaep_shim_preserves_exec_ctx_workspace() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="user-1",
        session_id="sess-1",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="tenant-A",
        workspace_id="ws-shim",
    )
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        agent_id="agent-1",
        workspace_id=request.workspace_id,
        request=request,
    )
    contract = AgentContract(id="agent-1", name="Agent", description="stub")
    step = AgentStep(step_index=0, step_id="s0", step_name="llm")

    class _StubAgent:
        pass

    step_ctx = build_step_context_from_uaep(_StubAgent(), step, exec_ctx)
    assembly = build_acp_assembly_request(step_ctx)
    assert assembly.workspace_id == "ws-shim"


def test_c2_workspace_independent_of_context_scope_id() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        tenant_id="tenant-A",
        workspace_id="ws-A",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        metadata={"context_scope_id": "ctx-77"},
    )
    assembly = build_acp_assembly_request(step_ctx)
    assert assembly.workspace_id == "ws-A"
    assert assembly.workspace_id != "ctx-77"


def test_c2_metadata_workspace_not_authoritative_in_production_builders() -> None:
    violations: list[str] = []
    for path in _AUTHORITY_PROPAGATION_PATHS:
        source = path.read_text(encoding="utf-8")
        if 'metadata.get("workspace_id")' in source or 'metadata["workspace_id"]' in source:
            violations.append(f"{path.relative_to(_REPO_ROOT)} uses metadata workspace authority")
    assert not violations, "\n".join(violations)


def test_c2_production_builders_set_explicit_workspace_id_kwarg() -> None:
    missing: list[str] = []
    for path in _PRODUCTION_BUILDER_PATHS:
        source = path.read_text(encoding="utf-8")
        if "ContextAssemblyRequest(" not in source:
            continue
        if "workspace_id=" not in source:
            missing.append(str(path.relative_to(_REPO_ROOT)))
    assert not missing, "\n".join(missing)


def test_c2_selection_only_without_workspace_still_valid() -> None:
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert request.workspace_id is None
    assert (
        resolve_ucl_artifact_ownership_scope(
            request,
            context_plan=SimpleNamespace(optimization_required=False),
        )
        is None
    )


def test_c2_artifact_required_missing_workspace_fails_closed() -> None:
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    assert (
        resolve_ucl_artifact_ownership_scope(
            request,
            context_plan=SimpleNamespace(optimization_required=True),
        )
        is None
    )


def test_c2_artifact_required_with_workspace_resolves_ownership() -> None:
    request = ContextAssemblyRequest(
        trace_id="t1",
        run_id="r1",
        task_id="task1",
        tenant_id="tenant1",
        workspace_id="ws-hot",
        assembly_scope="acp_step",
        objective="test",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(),
        assembly_options=TaskContextAssemblyOptions(),
    )
    scope = resolve_ucl_artifact_ownership_scope(
        request,
        context_plan=SimpleNamespace(optimization_required=True),
    )
    assert scope == UclArtifactOwnershipScope(tenant_id="tenant1", workspace_id="ws-hot")


def test_c2_iterative_tool_builder_uses_runtime_request_workspace_field() -> None:
    path = _PRODUCTION_BUILDER_PATHS[2]
    source = path.read_text(encoding="utf-8")
    assert "workspace_id=state.request.workspace_id" in source
    assert 'metadata.get("workspace_id")' not in source
