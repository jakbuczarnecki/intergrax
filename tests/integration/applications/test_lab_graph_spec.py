# © Artur Czarnecki. All rights reserved.

"""Lab graph spec validation and echo roster (Phase H-APP.3.6)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.graph_spec_to_plan import application_graph_spec_to_nexus_plan
from intergrax.applications.contracts.graph_spec import ApplicationGraphSpec, GraphNode
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent
from intergrax.applications.contracts.manifest import AgentBinding
from lab_application.manifest import build_lab_manifest_default

pytestmark = [pytest.mark.integration, pytest.mark.gate]


def test_lab_graph_spec_validates_echo_roster() -> None:
    manifest = build_lab_manifest_default()
    manifest = manifest.model_copy(
        update={
            "agents": [
                AgentBinding.mount(EchoAgent, capabilities=["echo.basic"]),
            ]
        }
    )
    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        edges=[],
    )
    spec.validate_against_roster(manifest.agents)


def test_lab_graph_spec_seeds_nexus_plan() -> None:
    spec = ApplicationGraphSpec(
        nodes=[GraphNode(agent_id="EchoAgent")],
        edges=[],
    )
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="graph seed",
        context=TaskContext(capability="echo.basic"),
    )
    plan = application_graph_spec_to_nexus_plan(
        spec,
        task,
        classification="single_agent_default",
    )
    assert len(plan.steps) == 1
    assert plan.steps[0].agent_id == "EchoAgent"
