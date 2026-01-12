# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Callable, Dict

from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.pipelines.no_planner_pipeline import NoPlannerPipeline
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep, build_runtime_step_registry
from intergrax.runtime.nexus.planning.step_executor_models import StepHandlerRegistry
from intergrax.runtime.nexus.planning.stepplan_models import StepAction
from intergrax.runtime.nexus.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.runtime_steps.rag_step import RagStep
from intergrax.runtime.nexus.runtime_steps.retrieve_attachments_step import RetrieveAttachmentsStep
from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.nexus.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from intergrax.runtime.nexus.runtime_steps.websearch_step import WebsearchStep


class PipelineFactory:

    @classmethod
    def build_pipeline(cls, state: RuntimeState) -> RuntimePipeline:             
        if state is None:
            raise ValueError("State is None.")
        if state.context is None:
            raise ValueError("state.context is None")
        if state.context.config is None:
            raise ValueError("state.context.config is None.")

        cfg = state.context.config

        # If you have config.validate(), keep it here for fail-fast consistency.
        # This should not introduce cyclic imports (config already exists on context).
        cfg.validate()

        # Explicit injection
        if cfg.pipeline is not None:
            return cfg.pipeline

        # Default pipeline (safe baseline)
        return NoPlannerPipeline()
    

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