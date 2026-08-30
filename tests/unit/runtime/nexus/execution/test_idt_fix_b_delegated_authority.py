# © Artur Czarnecki. All rights reserved.

"""IDT-FIX-B — delegated effective authority narrowing."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import pytest
from pydantic import ValidationError

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_execution_result import AgentExecutionStatus
from intergrax.contracts.delegation import DelegationSpec
from intergrax.contracts.delegation_authority import (
    EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY,
    EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY,
    EXECUTION_AUTHORITY_UNRESTRICTED_METADATA_KEY,
    EXECUTION_PERMISSION_SCOPES_METADATA_KEY,
    DelegationAuthorityError,
    EffectiveDelegationAuthority,
    ParentExecutionAuthority,
    mint_effective_delegation_authority,
    resolve_parent_execution_authority_for_node,
    validate_effective_delegation_metadata_assertions,
    validate_execution_authority_metadata_assertions,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.subtask_contract import SubtaskContract
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.interactions.actor_resolution import narrow_delegation_scopes
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.subagents.delegation_contract_enforcer import (
    DelegationToolPolicyError,
    enforce_subtask_tool_allowlist,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.uaep_gate_stubs import UaepPipelineStubAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@contextmanager
def _bound_execution_identity(
    *,
    authority: ParentExecutionAuthority | None = None,
) -> Iterator[None]:
    execution_id = mint_execution_id()
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        authority or ParentExecutionAuthority.unknown(),
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=create_execution_budget_ledger(None),
    )
    try:
        yield
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_authority(authority_token)
        reset_active_execution_identity(identity_token)


def test_child_cannot_expand_parent_authority() -> None:
    parent = ParentExecutionAuthority.scoped(("read",))
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=parent,
            requested_permission_scopes=("read", "write"),
        )


def test_normal_narrowing() -> None:
    parent = ParentExecutionAuthority.scoped(("read", "write"))
    effective = mint_effective_delegation_authority(
        parent=parent,
        requested_permission_scopes=("read",),
    )
    assert effective.effective_permission_scopes == ("read",)


def test_nested_delegation_cannot_recover_removed_scope() -> None:
    root = ParentExecutionAuthority.scoped(("read", "write", "admin"))
    child_effective = mint_effective_delegation_authority(
        parent=root,
        requested_permission_scopes=("read", "write"),
    )
    graph = ExecutionGraph(
        graph_id="nested",
        task_id="task_nested",
        nodes=[
            ExecutionNode(node_id="child", agent_id="child"),
            ExecutionNode(node_id="grandchild", agent_id="grandchild", depends_on=["child"]),
        ],
    )
    graph.node_by_id("child").metadata[EFFECTIVE_DELEGATION_AUTHORITY_NODE_KEY] = child_effective
    parent_for_grandchild = resolve_parent_execution_authority_for_node(
        graph,
        graph.node_by_id("grandchild"),
        root_authority=root,
    )
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=parent_for_grandchild,
            requested_permission_scopes=("admin",),
        )


def test_unknown_parent_does_not_mint_admin() -> None:
    with pytest.raises(DelegationAuthorityError):
        mint_effective_delegation_authority(
            parent=ParentExecutionAuthority.unknown(),
            requested_permission_scopes=("admin",),
        )


def test_empty_child_request_yields_empty_effective() -> None:
    parent = ParentExecutionAuthority.scoped(("read", "write"))
    effective = mint_effective_delegation_authority(
        parent=parent,
        requested_permission_scopes=(),
    )
    assert effective.effective_permission_scopes == ()


def test_tool_allowlist_positive_control_preserved() -> None:
    contract = SubtaskContract(child_agent_id="child", allowed_tools=("tool.a",))
    enforce_subtask_tool_allowlist(contract, "tool.a")
    with pytest.raises(DelegationToolPolicyError):
        enforce_subtask_tool_allowlist(contract, "tool.b")


def test_parent_tool_policy_preserved() -> None:
    contract = SubtaskContract(child_agent_id="child", allowed_tools=("tool.a", "tool.b"))
    with pytest.raises(DelegationToolPolicyError):
        enforce_subtask_tool_allowlist(
            contract,
            "tool.a",
            parent_allowed=("tool.b",),
        )


@pytest.mark.asyncio
async def test_effective_authority_propagated_to_child_request() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["read", "write"]},
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert child.last_metadata[EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY] == ["read"]
    assert child.last_request is not None
    assert isinstance(child.last_request.effective_delegation_authority, EffectiveDelegationAuthority)
    assert child.last_request.effective_delegation_authority.effective_permission_scopes == ("read",)


@pytest.mark.asyncio
async def test_delegation_granted_reports_effective_authority() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["read", "write"]},
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    bus = RuntimeEventBus(record_history=True)
    executor = GraphExecutor(registry, event_bus=bus)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    granted = [
        event
        for event in bus.history
        if event.event_type is RuntimeEventType.DELEGATION_GRANTED
    ]
    assert len(granted) == 1
    payload = granted[0].payload
    assert payload["requested_permission_scopes"] == ["read"]
    assert payload["effective_permission_scopes"] == ["read"]
    assert payload["permission_scopes"] == ["read"]


@pytest.mark.asyncio
async def test_rejected_delegation_does_not_execute_child() -> None:
    UaepPipelineStubAgent.run_log.clear()
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["read"]},
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read", "write"),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        executions, _, completed_graph, _ = await executor.execute(graph, task)
    assert UaepPipelineStubAgent.run_log == []
    node = completed_graph.node_by_id("n1")
    assert node.execution_result is not None
    assert node.execution_result.status is AgentExecutionStatus.FAILED
    assert executions == []


@pytest.mark.asyncio
async def test_rejected_delegation_does_not_emit_granted() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["read"]},
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("write",),
                ),
            ),
        ],
    )
    bus = RuntimeEventBus(record_history=True)
    executor = GraphExecutor(registry, event_bus=bus)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert not any(
        event.event_type is RuntimeEventType.DELEGATION_GRANTED for event in bus.history
    )


def test_idt_fix_a_actor_identity_remains_identity_only() -> None:
    actor = ActorIdentity(
        kind=ActorKind.USER,
        actor_id="u1",
        tenant_id="t1",
        permission_scopes=(),
    )
    delegation = DelegationSpec(child_agent_id="child", permission_scopes=("read",))
    with pytest.raises(DelegationAuthorityError):
        narrow_delegation_scopes(actor, delegation)


@pytest.mark.asyncio
async def test_a1_metadata_cannot_self_grant_root_scope() -> None:
    UaepPipelineStubAgent.run_log.clear()
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["admin"]},
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("admin",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity():
        executions, _, completed_graph, _ = await executor.execute(graph, task)
    assert UaepPipelineStubAgent.run_log == []
    node = completed_graph.node_by_id("n1")
    assert node.execution_result is not None
    assert node.execution_result.status is AgentExecutionStatus.FAILED
    assert executions == []


@pytest.mark.asyncio
async def test_a2_metadata_cannot_self_grant_unrestricted() -> None:
    UaepPipelineStubAgent.run_log.clear()
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        metadata={EXECUTION_AUTHORITY_UNRESTRICTED_METADATA_KEY: True},
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("admin",),
                ),
            ),
        ],
    )
    bus = RuntimeEventBus(record_history=True)
    executor = GraphExecutor(registry, event_bus=bus)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert UaepPipelineStubAgent.run_log == []
    assert not any(
        event.event_type is RuntimeEventType.DELEGATION_GRANTED for event in bus.history
    )


@pytest.mark.asyncio
async def test_a3_trusted_typed_scoped_root() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert child.last_request is not None
    assert child.last_request.effective_delegation_authority is not None
    assert child.last_request.effective_delegation_authority.effective_permission_scopes == ("read",)


@pytest.mark.asyncio
async def test_a4_trusted_typed_unrestricted_root() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.unrestricted_root(),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert child.last_request is not None
    assert child.last_request.effective_delegation_authority is not None
    assert child.last_request.effective_delegation_authority.effective_permission_scopes == ("read",)


@pytest.mark.asyncio
async def test_a5_conflicting_metadata_assertion_rejected() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.scoped(("read",)),
        metadata={EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["admin"]},
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        executions, _, _, _ = await executor.execute(graph, task)
    assert executions
    assert executions[0].status is AgentExecutionStatus.FAILED
    assert "conflict" in executions[0].errors[0]


def test_a5_metadata_conflict_validator() -> None:
    trusted = ParentExecutionAuthority.scoped(("read",))
    conflict = validate_execution_authority_metadata_assertions(
        {EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["admin"]},
        trusted,
    )
    assert conflict is not None


@pytest.mark.asyncio
async def test_b1_child_receives_typed_authority() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.child"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert child.last_request is not None
    authority = child.last_request.effective_delegation_authority
    assert isinstance(authority, EffectiveDelegationAuthority)
    assert authority.effective_permission_scopes == ("read",)


def test_b2_metadata_cannot_modify_effective_authority() -> None:
    trusted = EffectiveDelegationAuthority(
        requested_permission_scopes=("read",),
        parent_effective_scopes=("read", "write"),
        parent_unrestricted=False,
        effective_permission_scopes=("read",),
    )
    conflict = validate_effective_delegation_metadata_assertions(
        {EFFECTIVE_PERMISSION_SCOPES_METADATA_KEY: ["admin"]},
        trusted,
    )
    assert conflict is not None


@pytest.mark.asyncio
async def test_b3_graph_depends_on_does_not_create_execution_parent() -> None:
    """Graph depends_on is scheduling only; authority parent remains orchestration Execution."""
    UaepPipelineStubAgent.run_log.clear()
    parent = UaepPipelineStubAgent(
        agent_id="parent",
        capability="cap.parent",
        prefix="P",
    )
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    grandchild = UaepPipelineStubAgent(
        agent_id="grandchild",
        capability="cap.grandchild",
        prefix="G",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(parent)
    registry.register(child)
    registry.register(grandchild)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="go",
        context=TaskContext(capability="cap.parent"),
        execution_authority=ParentExecutionAuthority.scoped(("read", "write", "admin")),
    )
    graph = ExecutionGraph(
        graph_id="nested",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="parent",
                agent_id="parent",
                capability="cap.parent",
            ),
            ExecutionNode(
                node_id="child",
                agent_id="child",
                capability="cap.child",
                depends_on=["parent"],
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read", "write"),
                ),
            ),
            ExecutionNode(
                node_id="grandchild",
                agent_id="grandchild",
                capability="cap.grandchild",
                depends_on=["child"],
                delegation=DelegationSpec(
                    child_agent_id="grandchild",
                    permission_scopes=("admin",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        executions, _, completed_graph, _ = await executor.execute(graph, task)
    assert grandchild._agent_id in UaepPipelineStubAgent.run_log
    grandchild_node = completed_graph.node_by_id("grandchild")
    assert grandchild_node.execution_result is not None
    assert grandchild_node.execution_result.status is AgentExecutionStatus.COMPLETED
    assert grandchild.last_request is not None
    assert grandchild.last_request.effective_delegation_authority is not None
    assert grandchild.last_request.effective_delegation_authority.effective_permission_scopes == (
        "admin",
    )
    assert len(executions) == 3


def test_r2_1_public_task_envelope_cannot_accept_root_authority() -> None:
    with pytest.raises(ValidationError):
        TaskEnvelope(
            tenant_id="t",
            user_id="u",
            message="x",
            execution_authority={
                "permission_scopes": [],
                "unrestricted": True,
            },
        )


def test_r2_2_from_envelope_does_not_create_authority() -> None:
    envelope = TaskEnvelope(tenant_id="t", user_id="u", message="x")
    task = Task.from_envelope(envelope)
    assert task.execution_authority is None


@pytest.mark.asyncio
async def test_r2_3_caller_metadata_and_envelope_cannot_grant_authority() -> None:
    UaepPipelineStubAgent.run_log.clear()
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
    )
    registry = AgentRegistry()
    registry.register(child)
    envelope = TaskEnvelope(
        tenant_id="t1",
        user_id="u1",
        message="go",
        metadata={
            EXECUTION_PERMISSION_SCOPES_METADATA_KEY: ["admin"],
            EXECUTION_AUTHORITY_UNRESTRICTED_METADATA_KEY: True,
        },
    )
    task = Task.from_envelope(envelope)
    assert task.execution_authority is None
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("admin",),
                ),
            ),
        ],
    )
    bus = RuntimeEventBus(record_history=True)
    executor = GraphExecutor(registry, event_bus=bus)
    with _bound_execution_identity():
        executions, _, _, _ = await executor.execute(graph, task)
    assert UaepPipelineStubAgent.run_log == []
    assert not any(
        event.event_type is RuntimeEventType.DELEGATION_GRANTED for event in bus.history
    )
    assert executions
    assert executions[0].status is AgentExecutionStatus.FAILED
    assert "conflict" in executions[0].errors[0].lower()


@pytest.mark.asyncio
async def test_r2_4_trusted_host_enrichment_works() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    envelope = TaskEnvelope(tenant_id="t1", user_id="u1", message="go")
    task = Task.from_envelope(envelope).with_trusted_execution_authority(
        ParentExecutionAuthority.scoped(("read", "write"))
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert child.last_request is not None
    assert child.last_request.effective_delegation_authority is not None
    assert child.last_request.effective_delegation_authority.effective_permission_scopes == ("read",)


@pytest.mark.asyncio
async def test_r2_5_trusted_unrestricted_host_authority() -> None:
    child = UaepPipelineStubAgent(
        agent_id="child",
        capability="cap.child",
        prefix="C",
        track_request_metadata=True,
    )
    registry = AgentRegistry()
    registry.register(child)
    envelope = TaskEnvelope(tenant_id="t1", user_id="u1", message="go")
    task = Task.from_envelope(envelope).with_trusted_execution_authority(
        ParentExecutionAuthority.unrestricted_root()
    )
    graph = ExecutionGraph(
        graph_id="g1",
        task_id=task.task_id,
        nodes=[
            ExecutionNode(
                node_id="n1",
                agent_id="child",
                delegation=DelegationSpec(
                    child_agent_id="child",
                    permission_scopes=("read",),
                ),
            ),
        ],
    )
    executor = GraphExecutor(registry)
    with _bound_execution_identity(authority=task.execution_authority):
        await executor.execute(graph, task)
    assert child.last_request is not None
    assert child.last_request.effective_delegation_authority is not None
    assert child.last_request.effective_delegation_authority.effective_permission_scopes == ("read",)


def test_r2_6_public_envelope_cannot_set_effective_child_authority() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    trusted_child = EffectiveDelegationAuthority(
        requested_permission_scopes=("admin",),
        parent_effective_scopes=("admin",),
        parent_unrestricted=False,
        effective_permission_scopes=("admin",),
    )
    request = RuntimeRequest(
        agent_id="child",
        user_id="u1",
        session_id="s1",
        message="go",
        task_id=task_id,
        run_id=run_id,
        tenant_id="t1",
        execution_authority=ParentExecutionAuthority.scoped(("read", "write")),
        effective_delegation_authority=trusted_child,
    )
    envelope = request.to_envelope()
    restored = RuntimeRequest.from_envelope(envelope, task_id=task_id, run_id=run_id)
    assert restored.execution_authority is None
    assert restored.effective_delegation_authority is None
    with pytest.raises(ValidationError):
        TaskEnvelope(
            tenant_id="t1",
            user_id="u1",
            message="go",
            effective_delegation_authority=trusted_child.model_dump(),
        )
