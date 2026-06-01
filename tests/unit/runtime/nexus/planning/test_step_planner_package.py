# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.planning.step_planner import StepPlanner, StepPlannerConfig
from intergrax.runtime.nexus.planning.stepplan_models import EngineHints, PlanIntent, StepAction

pytestmark = pytest.mark.gate


def test_build_from_hints_generic_execute_plan_ends_with_finalize() -> None:
    planner = StepPlanner(StepPlannerConfig(max_total_steps=10))
    plan = planner.build_from_hints(
        user_message="Explain the harness policy bundle.",
        engine_hints=EngineHints(intent=PlanIntent.GENERIC),
        plan_id="test-plan",
    )
    assert plan.steps
    assert plan.steps[-1].action == StepAction.FINALIZE_ANSWER


def test_package_exports_config_and_planner() -> None:
    assert StepPlanner is not None
    assert StepPlannerConfig is not None
