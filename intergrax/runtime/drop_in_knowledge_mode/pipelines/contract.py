# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Dict
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep, build_runtime_step_registry
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import StepHandlerRegistry
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import StepAction
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.rag_step import RagStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.retrieve_attachments_step import RetrieveAttachmentsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.websearch_step import WebsearchStep

class RuntimePipeline(ABC):
    """
    Lightweight base class for pipeline runners.
    Shared implementation: execute_pipeline.
    """
    
    async def run(self, state: RuntimeState) -> RuntimeAnswer:
        """
        Public entrypoint. Provides shared validation and invariants.
        Subclasses must implement _inner_run().
        """
        self._validate_state(state)

        runtime_answer = await self._inner_run(state)

        if runtime_answer is None:
            raise RuntimeError("Pipeline returned None RuntimeAnswer.")

        # Hard invariant: Persist step (or equivalent) must set state.runtime_answer
        if state.runtime_answer is None:
            raise RuntimeError("Persist step did not set state.runtime_answer.")

        # Prefer the actual object returned by inner_run, but enforce consistency with state
        # (Optional) If you want to hard-enforce object identity:
        # if runtime_answer is not state.runtime_answer: ...
        self._assert_valid_answer(runtime_answer)

        return runtime_answer

    
    @abstractmethod
    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        """
        Pipeline-specific execution. Must produce and return RuntimeAnswer.
        """
        raise NotImplementedError
    

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


    def _validate_state(self, state: RuntimeState) -> None:
        if state is None:
            raise ValueError("state is None.")
        if state.context is None:
            raise ValueError("state.context is None.")
        if state.request is None:
            raise ValueError("state.request is None.")
        if not state.run_id:
            raise ValueError("state.run_id is missing.")
        

    async def _execute_pipeline(self, steps: list[RuntimeStep], state: RuntimeState) -> None:
        for step in steps:
            step_name = step.__class__.__name__
            state.trace_event(
                component="runtime",
                step=step_name,
                message="Step started",
                data={}
            )

            try:
                await step.run(state)
            except Exception as e:
                state.trace_event(
                    component="pipeline",
                    step=step_name,
                    message="Step failed",
                    data={"error": repr(e)},
                )
                raise

            state.trace_event(
                component="runtime",
                step=step_name,
                message="Step finished",
                data={}
            )

    def _assert_valid_answer(self, answer: RuntimeAnswer) -> None:
        assert answer is not None, "Pipeline returned None answer"
        assert answer.route is not None, "RuntimeAnswer has no route"
