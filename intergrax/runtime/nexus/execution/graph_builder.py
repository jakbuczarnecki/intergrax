# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.planning.task_planner import NexusPlan


def plan_to_execution_graph(plan: NexusPlan) -> ExecutionGraph:
    """Convert a NexusPlan into an ExecutionGraph."""
    nodes = [
        ExecutionNode(
            node_id=step.step_id,
            agent_id=step.agent_id,
            capability=step.capability,
            description=step.description,
            depends_on=list(step.depends_on),
        )
        for step in plan.steps
    ]
    return ExecutionGraph(
        graph_id=plan.plan_id,
        task_id=plan.task_id,
        nodes=nodes,
    )
