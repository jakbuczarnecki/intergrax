# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.drop_in_knowledge_mode.config import StepPlanningStrategy
from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.no_planner_pipeline import NoPlannerPipeline
from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.planner_dynamic_pipeline import PlannerDynamicPipeline
from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.planner_static_pipeline import PlannerStaticPipeline
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_context import RuntimeContext
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState


class PipelineFactory:

    @classmethod
    def create(cls, state: RuntimeState) -> RuntimePipeline:

        if state.context.config.step_planning_strategy == StepPlanningStrategy.OFF:
            return NoPlannerPipeline()
        
        if state.context.config.step_planning_strategy == StepPlanningStrategy.STATIC_PLAN:
            return PlannerStaticPipeline()
        
        if state.context.config.step_planning_strategy == StepPlanningStrategy.DYNAMIC_LOOP:
            return PlannerDynamicPipeline()
        
        raise ValueError(f"Unknown step_planning_strategy: {state.context.config.step_planning_strategy}")