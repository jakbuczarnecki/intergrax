# © Artur Czarnecki. All rights reserved.

"""ApplicationGraphSpec to NexusPlan conversion (Phase ORCH-2)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.graph_spec_to_plan import (
    application_graph_spec_to_nexus_plan,
    should_seed_plan_from_graph_spec,
)
from intergrax.applications.contracts.graph_spec import (
    ApplicationGraphSpec,
    GraphEdge,
    GraphEdgeKind,
    GraphNode,
)
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _task() -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability="echo.basic"),
    )


def test_should_seed_when_no_plan_id() -> None:
    task = _task()
    assert should_seed_plan_from_graph_spec(task) is True
    task.runtime.orchestration.plan_id = "plan_existing"
    task.sync_metadata()
    assert should_seed_plan_from_graph_spec(task) is False


def test_graph_spec_to_plan_topology() -> None:
    spec = ApplicationGraphSpec(
        nodes=[
            GraphNode(agent_id="AgentA"),
            GraphNode(agent_id="AgentB"),
        ],
        edges=[
            GraphEdge(
                source_agent_id="AgentA",
                target_agent_id="AgentB",
                kind=GraphEdgeKind.DEPENDS_ON,
            ),
            GraphEdge(
                source_agent_id="AgentA",
                target_agent_id="AgentB",
                kind=GraphEdgeKind.DELEGATES_TO,
            ),
        ],
    )
    plan = application_graph_spec_to_nexus_plan(
        spec,
        _task(),
        classification="multi_agent",
    )
    assert len(plan.steps) == 2
    step_a = next(s for s in plan.steps if s.agent_id == "AgentA")
    step_b = next(s for s in plan.steps if s.agent_id == "AgentB")
    assert step_b.depends_on == [step_a.step_id]
    assert step_a.delegation is not None
    assert step_a.delegation.child_agent_id == "AgentB"
