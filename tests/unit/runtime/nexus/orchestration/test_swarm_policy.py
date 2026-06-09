# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.runtime.nexus.orchestration.swarm_policy import (
    SwarmCoordinationError,
    annotate_plan_coordination_pattern,
    validate_swarm_parallel_batch,
    validate_swarm_plan,
)
from intergrax.runtime.nexus.planning.task_planner import NexusPlan, PlanStep

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_validate_swarm_plan_requires_three_parallel_roots() -> None:
    plan = NexusPlan(
        task_id="t1",
        classification="multi_agent",
        steps=[
            PlanStep(step_id="a", agent_id="a1"),
            PlanStep(step_id="b", agent_id="a2"),
        ],
    )
    with pytest.raises(SwarmCoordinationError):
        validate_swarm_plan(plan)


def test_annotate_plan_coordination_pattern_swarm() -> None:
    plan = NexusPlan(
        task_id="t1",
        classification="multi_agent",
        steps=[
            PlanStep(step_id="a", agent_id="a1"),
            PlanStep(step_id="b", agent_id="a2"),
            PlanStep(step_id="c", agent_id="a3"),
        ],
    )
    annotated = annotate_plan_coordination_pattern(
        plan,
        coordination_pattern=CoordinationPattern.SWARM.value,
    )
    assert annotated.plan_metadata["coordination_pattern"] == CoordinationPattern.SWARM.value


def test_validate_swarm_parallel_batch_requires_three_nodes() -> None:
    with pytest.raises(SwarmCoordinationError):
        validate_swarm_parallel_batch(2)
