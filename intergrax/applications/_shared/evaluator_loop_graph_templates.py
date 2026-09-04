# © Artur Czarnecki. All rights reserved.

"""Evaluator-loop standard graph templates for product hosts (AUDIT-IDEAL-10.1)."""

from __future__ import annotations

from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    EvaluatorLoopGraphBinding,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)
from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.runtime.nexus.execution.evaluator_loop_spec import EvaluatorLoopSpec


def evaluator_loop_graph_template(
    *,
    producer_agent_id: str,
    evaluator_agent_id: str,
    revise_agent_id: str,
    max_iterations: int = 2,
    min_score: float = 0.75,
) -> ApplicationGraphSpec:
    """Producer → evaluator critique loop with optional revise routing."""
    revise_step = f"node_{revise_agent_id}"
    spec = EvaluatorLoopSpec(
        max_iterations=max_iterations,
        min_score=min_score,
        revise_node_id=revise_step,
    )
    return ApplicationGraphSpec(
        nodes=[
            GraphNode(agent_id=producer_agent_id),
            GraphNode(agent_id=evaluator_agent_id),
            GraphNode(agent_id=revise_agent_id),
        ],
        edges=[
            GraphEdge(
                source_agent_id=producer_agent_id,
                target_agent_id=evaluator_agent_id,
                kind=GraphEdgeKind.DEPENDS_ON,
            ),
            GraphEdge(
                source_agent_id=evaluator_agent_id,
                target_agent_id=revise_agent_id,
                kind=GraphEdgeKind.DEPENDS_ON,
            ),
        ],
        trigger_capabilities=[f"{CoordinationPattern.EVALUATOR_LOOP.value}.pipeline"],
        evaluator_loop=EvaluatorLoopGraphBinding(
            producer_agent_id=producer_agent_id,
            evaluator_agent_id=evaluator_agent_id,
            revise_agent_id=revise_agent_id,
            spec=spec,
        ),
    )
