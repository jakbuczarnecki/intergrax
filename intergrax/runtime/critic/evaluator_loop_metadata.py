# © Artur Czarnecki. All rights reserved.

"""Evaluator-loop graph metadata keys (Phase CRIT-V-4.2)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode

COORDINATION_PATTERN_KEY = "coordination_pattern"
EVALUATOR_LOOP_SPEC_KEY = "evaluator_loop_spec"
EVALUATOR_LOOP_ITERATION_KEY = "evaluator_loop_iteration"


def evaluator_loop_spec_from_node(node: ExecutionNode) -> EvaluatorLoopSpec | None:
    """Return loop spec when node is tagged with ``EVALUATOR_LOOP`` coordination."""
    pattern = node.metadata.get(COORDINATION_PATTERN_KEY)
    if pattern != CoordinationPattern.EVALUATOR_LOOP.value:
        return None
    raw = node.metadata.get(EVALUATOR_LOOP_SPEC_KEY)
    if isinstance(raw, EvaluatorLoopSpec):
        return raw
    if isinstance(raw, dict):
        return EvaluatorLoopSpec.model_validate(raw)
    return None


def tag_node_evaluator_loop(node: ExecutionNode, spec: EvaluatorLoopSpec) -> None:
    """Attach evaluator-loop coordination metadata to a graph node."""
    node.metadata[COORDINATION_PATTERN_KEY] = CoordinationPattern.EVALUATOR_LOOP.value
    node.metadata[EVALUATOR_LOOP_SPEC_KEY] = spec.model_dump()


def current_evaluator_loop_iteration(node: ExecutionNode) -> int:
    value = node.metadata.get(EVALUATOR_LOOP_ITERATION_KEY, 0)
    return int(value) if isinstance(value, int) else 0


def set_evaluator_loop_iteration(node: ExecutionNode, iteration: int) -> None:
    node.metadata[EVALUATOR_LOOP_ITERATION_KEY] = iteration


def critic_feedback_context(verdict_errors: list[str]) -> dict[str, Any]:
    """Context payload passed to revise-node execution."""
    return {"critic_feedback": list(verdict_errors)}
