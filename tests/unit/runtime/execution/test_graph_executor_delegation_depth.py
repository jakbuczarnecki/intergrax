# © Artur Czarnecki. All rights reserved.

"""Delegation depth and budget enforcement (Phase FLOW-3/15)."""

from __future__ import annotations

import pytest

from intergrax.contracts.delegation import DelegationSpec
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_delegation_depth_limit() -> None:
    registry = AgentRegistry()
    executor = GraphExecutor(registry, max_delegation_depth=1)
    graph = ExecutionGraph(
        graph_id="g1",
        task_id="task_depth",
        nodes=[
            ExecutionNode(node_id="parent", agent_id="parent"),
            ExecutionNode(
                node_id="child",
                agent_id="child",
                depends_on=["parent"],
                delegation=DelegationSpec(child_agent_id="child", parent_node_id="parent"),
            ),
            ExecutionNode(
                node_id="grandchild",
                agent_id="grandchild",
                depends_on=["child"],
                delegation=DelegationSpec(
                    child_agent_id="grandchild",
                    parent_node_id="child",
                ),
            ),
        ],
    )
    error = executor._validate_delegation_constraints(graph, graph.node_by_id("grandchild"))
    assert error is not None
    assert "max_delegation_depth" in error


def test_delegation_budget_limit() -> None:
    registry = AgentRegistry()
    executor = GraphExecutor(registry)
    graph = ExecutionGraph(
        graph_id="g1",
        task_id="task_budget",
        nodes=[
            ExecutionNode(
                node_id="child",
                agent_id="child",
                delegation=DelegationSpec(child_agent_id="child", max_llm_calls=0),
            ),
        ],
    )
    error = executor._validate_delegation_constraints(graph, graph.nodes[0])
    assert error == "delegation_budget_llm_calls_exhausted"
