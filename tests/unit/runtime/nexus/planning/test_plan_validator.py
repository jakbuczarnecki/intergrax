# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.nexus.planning.plan_validator import validate_nexus_plan
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep
from intergrax.runtime.registry.agent_registry import AgentRegistry
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_validate_nexus_plan_rejects_unknown_depends_on() -> None:
    plan = NexusPlan(
        task_id="t1",
        classification="multi_agent",
        steps=[
            PlanStep(step_id="a", agent_id="agent_a"),
            PlanStep(step_id="b", agent_id="agent_b", depends_on=["missing"]),
        ],
    )
    registry = AgentRegistry()

    class _AgentA(EchoAgent):
        def get_contract(self):
            contract = super().get_contract()
            contract.id = "agent_a"
            return contract

    class _AgentB(EchoAgent):
        def get_contract(self):
            contract = super().get_contract()
            contract.id = "agent_b"
            return contract

    registry.register(_AgentA())
    registry.register(_AgentB())
    errors = validate_nexus_plan(plan, registry)
    assert any("unknown depends_on" in err for err in errors)


def test_validate_nexus_plan_rejects_unknown_agent_id() -> None:
    plan = NexusPlan(
        task_id="t1",
        classification="single_agent",
        steps=[PlanStep(step_id="a", agent_id="ghost")],
    )
    registry = AgentRegistry()
    errors = validate_nexus_plan(plan, registry)
    assert any("unknown agent_id" in err for err in errors)
