# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Core runtime engine for Drop-In Knowledge Mode.

This module defines the `DropInKnowledgeRuntime` class, which:
  - loads or creates chat sessions,
  - appends user messages,
  - builds a conversation history for the LLM,
  - augments context with RAG, web search and tools,
  - produces a `RuntimeAnswer` object as a high-level response.

The goal is to provide a single, simple entrypoint that can be used from
FastAPI, Streamlit, MCP-like environments, CLI tools, etc.

Refactored as a stateful pipeline:

  - RuntimeState holds all intermediate data (session, history, flags, debug).
  - Each step mutates the state and can be inspected in isolation.
  - ask() just wires the steps together in a readable order.
"""

from __future__ import annotations

import asyncio
import uuid

from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.drop_in_knowledge_mode.config import StepPlanningStrategy
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_context import RuntimeContext
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.build_base_history_step import BuildBaseHistoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.contract import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.history_step import HistoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.ingested_attachments_step import IngestedAttachmentsStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.instructions_step import InstructionsStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.profile_based_memory_step import ProfileBasedMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.rag_step import RagStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.session_and_ingest_step import SessionAndIngestStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.user_longterm_memory_step import UserLongtermMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.websearch_step import WebsearchStep
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
)
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState


# ----------------------------------------------------------------------
# DropInKnowledgeRuntime
# ----------------------------------------------------------------------


class DropInKnowledgeRuntime:
    """
    High-level conversational runtime for the Intergrax framework.

    This class is designed to behave like a ChatGPT/Claude-style engine,
    but fully powered by Intergrax components (LLM adapters, RAG, web search,
    tools, memory, etc.).

    Responsibilities (current stage):
      - Accept a RuntimeRequest.
      - Load or create a ChatSession via SessionManager.
      - Append the user message to the session.
      - Build an LLM-ready context:
          * system prompt(s),
          * chat history from SessionManager,
          * optional retrieved chunks from documents (RAG),
          * optional web search context (if enabled),
          * optional tools results.
      - Call the main LLM adapter once with the fully enriched context
        to produce the final answer.
      - Append the assistant message to the session.
      - Return a RuntimeAnswer with the final answer text and metadata.
    """

    def __init__(
        self,
        context: RuntimeContext
    ) -> None:
        self.context = context


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.
        """

        run_id = f"run_{uuid.uuid4().hex}"

        state = RuntimeState(
            context=self.context,
            request=request,
            run_id=run_id,
            llm_usage_tracker=LLMUsageTracker(run_id=run_id)
        )        
        
        state.configure_llm_tracker()

        # Initial trace entry for this request.
        state.trace_event(
            component="engine",
            step="run_start",
            message="DropInKnowledgeRuntime.run() called.",
            data={
                "session_id": request.session_id,
                "user_id": request.user_id,
                "tenant_id": request.tenant_id or self.context.config.tenant_id,
                "run_id": state.run_id,
                "step_planning_strategy": str(self.context.config.step_planning_strategy),
            },
        )

        if self.context.config.step_planning_strategy == StepPlanningStrategy.OFF:
            runtime_answer = await self._run_pipeline_no_planner(state=state)
        
        elif self.context.config.step_planning_strategy == StepPlanningStrategy.STATIC_PLAN:
            runtime_answer = await self._run_pipeline_static_plan(state)
        
        elif self.context.config.step_planning_strategy == StepPlanningStrategy.DYNAMIC_LOOP:
            runtime_answer = await self._run_pipeline_dynamic_loop(state)

        else:
            raise ValueError(f"Unknown step_planning_strategy: {self.context.config.step_planning_strategy}")


        # Final trace entry for this request.
        state.trace_event(
            component="engine",
            step="run_end",
            message="DropInKnowledgeRuntime.run() finished.",
            data={
                "strategy": runtime_answer.route.strategy,
                "used_rag": runtime_answer.route.used_rag,
                "used_websearch": runtime_answer.route.used_websearch,
                "used_tools": runtime_answer.route.used_tools,
                "used_user_longterm_memory": runtime_answer.route.used_user_longterm_memory,
                "run_id":state.run_id
            },
        )
        
        await state.finalize_llm_tracker(request, runtime_answer)

        return runtime_answer


    async def _run_pipeline_no_planner(self, state: RuntimeState) -> RuntimeAnswer:
        
        pipeline = [
            SessionAndIngestStep(),
            ProfileBasedMemoryStep(),
            BuildBaseHistoryStep(),
            HistoryStep(),
            InstructionsStep(),

            RagStep(),
            UserLongtermMemoryStep(),
            IngestedAttachmentsStep(),
            WebsearchStep(),

            EnsureCurrentUserMessageStep(),

            ToolsStep(),
            CoreLLMStep(),
            PersistAndBuildAnswerStep(),
        ]

        await self._execute_pipeline(pipeline, state)

        runtime_answer = state.runtime_answer
        if runtime_answer is None:
            raise RuntimeError("Persist step did not set state.runtime_answer.")
        
        return runtime_answer


    async def _run_pipeline_static_plan(self, state: RuntimeState) -> RuntimeAnswer:
        raise NotImplementedError(
            "StepPlanningStrategy.STATIC_PLAN is configured, but step planner is not implemented in this session."
        )

    async def _run_pipeline_dynamic_loop(self, state: RuntimeState) -> RuntimeAnswer:
        raise NotImplementedError(
            "StepPlanningStrategy.DYNAMIC_LOOP is configured, but step planner is not implemented in this session."
        )
    

    async def _execute_pipeline(self, steps: list[RuntimeStep], state: RuntimeState) -> None:
        for step in steps:
            step_name = step.__class__.__name__
            state.trace_event(
                component="runtime",
                step=step_name,
                message="Step started",
                data={}
            )

            await step.run(state)

            state.trace_event(
                component="runtime",
                step=step_name,
                message="Step finished",
                data={}
            )


    def run_sync(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Synchronous wrapper around `run()`.

        Useful for environments where `await` is not easily available,
        such as simple scripts or some notebook setups.
        """
        return asyncio.run(self.run(request))
    