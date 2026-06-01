# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan
from intergrax.runtime.nexus.planning.stepplan_models import (
    EngineHints,
    ExecutionPlan,
    PlanBuildMode,
)

from intergrax.runtime.nexus.planning.step_planner.assembly import StepPlanAssembly
from intergrax.runtime.nexus.planning.step_planner.config import StepPlannerConfig
from intergrax.runtime.nexus.planning.step_planner.step_factory import StepPlanStepFactory
from intergrax.runtime.nexus.planning.step_planner.strategies import StepPlanStrategies


class StepPlanner:
    """
    Deterministic planner that builds an ExecutionPlan from:
      - user_message
      - engine_hints (e.g. from engine_planner.py)

    Important: ExecutionStep.params MUST be a dict (per stepplan_models.ExecutionStep).
    """

    def __init__(self, cfg: Optional[StepPlannerConfig] = None) -> None:
        self._cfg = cfg or StepPlannerConfig()
        self._factory = StepPlanStepFactory(self._cfg)
        self._assembly = StepPlanAssembly(self._cfg, self._factory)
        self._strategies = StepPlanStrategies(self._cfg, self._factory, self._assembly)

    def build_from_engine_plan(
        self,
        *,
        user_message: str,
        engine_plan: EnginePlan,
        plan_id: Optional[str] = None,
        build_mode: PlanBuildMode = PlanBuildMode.STATIC,
    ) -> ExecutionPlan:
        return self._strategies.build_from_engine_plan(
            user_message=user_message,
            engine_plan=engine_plan,
            plan_id=plan_id,
            build_mode=build_mode,
        )

    def build_from_hints(
        self,
        *,
        user_message: str,
        engine_hints: Optional[EngineHints] = None,
        plan_id: Optional[str] = None,
    ) -> ExecutionPlan:
        return self._strategies.build_from_hints(
            user_message=user_message,
            engine_hints=engine_hints,
            plan_id=plan_id,
        )
