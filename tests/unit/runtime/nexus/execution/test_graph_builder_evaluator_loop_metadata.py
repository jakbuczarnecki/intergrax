# © Artur Czarnecki. All rights reserved.

"""Generic plan→graph evaluator-loop metadata handling (PLATFORM-5A)."""

from __future__ import annotations

import pytest

from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    EvaluatorLoopGraphBinding,
    GraphNode,
)
from intergrax.runtime.critic.evaluator_loop_metadata import evaluator_loop_spec_from_node
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.nexus.execution.graph_builder import plan_to_execution_graph
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PRODUCER_AGENT_ID = "echo_agent"


def _plan_with_evaluator_metadata(
    *,
    metadata_value: str,
    producer_agent_id: str = _PRODUCER_AGENT_ID,
) -> NexusPlan:
    return NexusPlan(
        plan_id="plan_evaluator_loop",
        task_id="task_evaluator_loop",
        classification="multi_agent",
        steps=[
            PlanStep(
                step_id=f"node_{producer_agent_id}",
                agent_id=producer_agent_id,
                capability="echo.basic",
                description="producer",
            )
        ],
        plan_metadata={"evaluator_loop.v1": metadata_value},
    )


def test_plan_to_execution_graph_tags_producer_when_binding_valid() -> None:
    binding = EvaluatorLoopGraphBinding(
        producer_agent_id=_PRODUCER_AGENT_ID,
        evaluator_agent_id=_PRODUCER_AGENT_ID,
        revise_agent_id=_PRODUCER_AGENT_ID,
        spec=EvaluatorLoopSpec(
            max_iterations=3,
            revise_node_id=f"node_{_PRODUCER_AGENT_ID}",
        ),
    )
    graph = plan_to_execution_graph(
        _plan_with_evaluator_metadata(metadata_value=binding.model_dump_json())
    )
    producer = graph.nodes[0]
    spec = evaluator_loop_spec_from_node(producer)
    assert spec is not None
    assert spec.max_iterations == 3


def test_plan_to_execution_graph_ignores_malformed_evaluator_metadata() -> None:
    graph = plan_to_execution_graph(_plan_with_evaluator_metadata(metadata_value="{not-json"))
    assert evaluator_loop_spec_from_node(graph.nodes[0]) is None


def test_plan_to_execution_graph_ignores_missing_producer_node() -> None:
    binding = EvaluatorLoopGraphBinding(
        producer_agent_id="missing_agent",
        evaluator_agent_id=_PRODUCER_AGENT_ID,
        revise_agent_id=_PRODUCER_AGENT_ID,
        spec=EvaluatorLoopSpec(
            max_iterations=2,
            revise_node_id=f"node_{_PRODUCER_AGENT_ID}",
        ),
    )
    graph = plan_to_execution_graph(
        _plan_with_evaluator_metadata(metadata_value=binding.model_dump_json())
    )
    assert evaluator_loop_spec_from_node(graph.nodes[0]) is None


def test_plan_to_execution_graph_without_evaluator_metadata_preserves_legacy_graph() -> None:
    plan = NexusPlan(
        plan_id="plan_plain",
        task_id="task_plain",
        classification="multi_agent",
        steps=[
            PlanStep(
                step_id=f"node_{_PRODUCER_AGENT_ID}",
                agent_id=_PRODUCER_AGENT_ID,
                capability="echo.basic",
                description="producer",
            )
        ],
    )
    graph = plan_to_execution_graph(plan)
    assert evaluator_loop_spec_from_node(graph.nodes[0]) is None


def test_application_graph_spec_serializes_evaluator_binding_for_planner() -> None:
    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id=_PRODUCER_AGENT_ID)],
        trigger_capabilities=["echo.basic"],
        evaluator_loop=EvaluatorLoopGraphBinding(
            producer_agent_id=_PRODUCER_AGENT_ID,
            evaluator_agent_id=_PRODUCER_AGENT_ID,
            revise_agent_id=_PRODUCER_AGENT_ID,
            spec=EvaluatorLoopSpec(
                max_iterations=2,
                revise_node_id=f"node_{_PRODUCER_AGENT_ID}",
            ),
        ),
    )
    assert spec.evaluator_loop is not None
    assert spec.evaluator_loop.spec.max_iterations == 2
