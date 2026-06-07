# © Artur Czarnecki. All rights reserved.

"""Multi-agent ordering policy (Phase FLOW-17)."""

from __future__ import annotations

import pytest

from intergrax.contracts.orchestration_enums import MultiAgentOrder
from intergrax.runtime.nexus.planning.task_planner import TaskPlanner
from intergrax.runtime.nexus.task_classifier import TaskClassification
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_stable_alpha_orders_agents() -> None:
    registry = AgentRegistry()

    class _ZAgent(EchoAgent):
        def get_contract(self):
            contract = super().get_contract()
            contract.id = "z_agent"
            return contract

    class _AAgent(EchoAgent):
        def get_contract(self):
            contract = super().get_contract()
            contract.id = "a_agent"
            return contract

    registry.register(_ZAgent())
    registry.register(_AAgent())

    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        context=TaskContext(capability="echo.basic"),
        classification=TaskClassification.MULTI_AGENT.value,
    )
    planner = TaskPlanner(multi_agent_order=MultiAgentOrder.STABLE_ALPHA)
    plan = planner.plan(task, registry)
    agent_ids = [step.agent_id for step in plan.steps if step.agent_id]
    assert agent_ids == sorted(agent_ids)
