# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Callable, Dict

from intergrax.runtime.drop_in_knowledge_mode.config import StepPlanningStrategy
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.pipelines.no_planner_pipeline import NoPlannerPipeline
from intergrax.runtime.drop_in_knowledge_mode.pipelines.planner_dynamic_pipeline import PlannerDynamicPipeline
from intergrax.runtime.drop_in_knowledge_mode.pipelines.planner_static_pipeline import PlannerStaticPipeline
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep, build_runtime_step_registry
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import StepHandlerRegistry
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import StepAction
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.rag_step import RagStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.retrieve_attachments_step import RetrieveAttachmentsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.websearch_step import WebsearchStep


class PipelineFactory:

    @classmethod
    def build_pipeline(cls, state: RuntimeState) -> RuntimePipeline:

        if state.context.config.step_planning_strategy == StepPlanningStrategy.OFF:
            return NoPlannerPipeline()
        
        if state.context.config.step_planning_strategy == StepPlanningStrategy.STATIC_PLAN:
            return PlannerStaticPipeline()
        
        if state.context.config.step_planning_strategy == StepPlanningStrategy.DYNAMIC_LOOP:
            return PlannerDynamicPipeline()
        
        raise ValueError(f"Unknown step_planning_strategy: {state.context.config.step_planning_strategy}")
    

    @classmethod
    def build_default_planning_step_registry(cls) -> StepHandlerRegistry:
        """
        Default registry for StepExecutor planning actions (STATIC/DYNAMIC).
        This registry is explicit and production-safe (no reflection).
        """
        bindings: Dict[StepAction, Callable[[], RuntimeStep]] = {
            StepAction.USE_WEBSEARCH: lambda: WebsearchStep(),
            StepAction.USE_TOOLS: lambda: ToolsStep(),
            StepAction.USE_RAG_RETRIEVAL: lambda: RagStep(),
            StepAction.USE_ATTACHMENTS_RETRIEVAL: lambda: RetrieveAttachmentsStep(),
            StepAction.USE_USER_LONGTERM_MEMORY_SEARCH: lambda: UserLongtermMemoryStep(),
            StepAction.SYNTHESIZE_DRAFT: lambda: CoreLLMStep(),
            StepAction.VERIFY_ANSWER: lambda: CoreLLMStep(),
            StepAction.FINALIZE_ANSWER: lambda: PersistAndBuildAnswerStep(),            
        }

        return build_runtime_step_registry(bindings=bindings)