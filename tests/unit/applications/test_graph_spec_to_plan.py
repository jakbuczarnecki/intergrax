# © Artur Czarnecki. All rights reserved.

"""ApplicationGraphSpec to NexusPlan conversion (Phase ORCH-2, ORCH-CONFIG.2)."""

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


def _task(*, capability: str = "echo.pipeline") -> Task:
    return Task(
        tenant_id="t1",
        user_id="u1",
        message="hello",
        context=TaskContext(capability=capability),
    )


def _spec(*, triggers: list[str] | None = None) -> ApplicationGraphSpec:
    return ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="AgentA"), GraphNode(agent_id="AgentB")],
        edges=[
            GraphEdge(
                source_agent_id="AgentA",
                target_agent_id="AgentB",
                kind=GraphEdgeKind.DEPENDS_ON,
            ),
        ],
        trigger_capabilities=triggers or [],
    )


def test_should_seed_when_no_plan_id_and_pipeline_capability() -> None:
    task = _task(capability="echo.pipeline")
    assert should_seed_plan_from_graph_spec(task, _spec()) is True


def test_should_not_seed_for_non_pipeline_capability_without_triggers() -> None:
    task = _task(capability="echo.basic")
    assert should_seed_plan_from_graph_spec(task, _spec()) is False


def test_should_seed_for_explicit_trigger_capabilities() -> None:
    task = _task(capability="echo.basic")
    spec = _spec(triggers=["echo.basic"])
    assert should_seed_plan_from_graph_spec(task, spec) is True


def test_should_not_seed_when_plan_id_present() -> None:
    task = _task(capability="echo.pipeline")
    task.runtime.orchestration.plan_id = "plan_existing"
    task.sync_metadata()
    assert should_seed_plan_from_graph_spec(task, _spec()) is False


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
    assert step_a.delegation is None
    assert step_b.delegation is not None
    assert step_b.delegation.child_agent_id == "AgentB"
    assert step_b.delegation.parent_node_id == step_a.step_id
    assert step_b.delegation.inherit_tool_policy is False
