# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.applications.contracts.graph_spec import EvaluatorLoopGraphBinding
from intergrax.runtime.critic.evaluator_loop_metadata import tag_node_evaluator_loop
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode
from intergrax.runtime.nexus.planning.task_planner import NexusPlan

_EVALUATOR_LOOP_PLAN_METADATA_KEY = "evaluator_loop.v1"


def _step_id_for_agent(agent_id: str) -> str:
    return f"node_{agent_id}"


def _apply_evaluator_loop_metadata(plan: NexusPlan, nodes: list[ExecutionNode]) -> None:
    """Attach evaluator-loop coordination from graph-spec plan metadata (ORCH-2)."""
    raw = plan.plan_metadata.get(_EVALUATOR_LOOP_PLAN_METADATA_KEY)
    if not raw:
        return
    binding = EvaluatorLoopGraphBinding.model_validate_json(raw)
    producer_step_id = _step_id_for_agent(binding.producer_agent_id)
    for node in nodes:
        if node.node_id == producer_step_id:
            tag_node_evaluator_loop(node, binding.spec)
            break


def plan_to_execution_graph(plan: NexusPlan) -> ExecutionGraph:
    """Convert a NexusPlan into an ExecutionGraph."""
    nodes = [
        ExecutionNode(
            node_id=step.step_id,
            agent_id=step.agent_id,
            capability=step.capability,
            description=step.description,
            depends_on=list(step.depends_on),
            delegation=step.delegation,
        )
        for step in plan.steps
    ]
    _apply_evaluator_loop_metadata(plan, nodes)
    return ExecutionGraph(
        graph_id=plan.plan_id,
        task_id=plan.task_id,
        nodes=nodes,
    )
