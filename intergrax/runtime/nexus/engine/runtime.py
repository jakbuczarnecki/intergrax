# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Core runtime engine for nexus Mode.

This module defines the `RuntimeEngine` class, which:
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

import uuid

from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata
from intergrax.runtime.nexus.tracing.runtime.runtime_run_abort import RuntimeRunAbortDiagV1
from intergrax.runtime.nexus.tracing.runtime.runtime_run_end import RuntimeRunEndDiagV1
from intergrax.runtime.nexus.tracing.runtime.runtime_run_start import RuntimeRunStartDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ----------------------------------------------------------------------
# RuntimeEngine
# ----------------------------------------------------------------------


class RuntimeEngine:
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
            llm_usage_tracker=LLMUsageTracker(run_id=run_id),
        )

        state.configure_llm_tracker()

        pipeline = PipelineFactory.build_pipeline(state=state)

        # Initial trace entry for this request.
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="run_start",
            level=TraceLevel.INFO,
            message="RuntimeEngine.run() called.",
            payload=RuntimeRunStartDiagV1(
                session_id=request.session_id,
                user_id=request.user_id,
                tenant_id=(request.tenant_id or self.context.config.tenant_id),
                run_id=state.run_id,
                pipeline_name=pipeline.__class__.__name__ if pipeline is not None else "None",
            ),
        )

        runtime_answer: RuntimeAnswer | None = None

        try:
            
            runtime_answer = await pipeline.run(state=state)           

            # Final trace entry for this request.
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="run_end",
                level=TraceLevel.INFO,
                message="RuntimeEngine.run() finished.",
                payload=RuntimeRunEndDiagV1(
                    strategy=runtime_answer.route.strategy,
                    used_rag=runtime_answer.route.used_rag,
                    used_websearch=runtime_answer.route.used_websearch,
                    used_tools=runtime_answer.route.used_tools,
                    used_user_longterm_memory=runtime_answer.route.used_user_longterm_memory,
                    run_id=state.run_id,
                ),
            )

            return runtime_answer

        finally:
            await state.finalize_llm_tracker(
                request=request,
                runtime_answer=runtime_answer,
            )

            state.trace_event(
                component=TraceComponent.ENGINE,
                step="run_abort",
                level=TraceLevel.WARNING,
                message="RuntimeEngine.run() aborted before RuntimeAnswer was produced.",
                payload=RuntimeRunAbortDiagV1(run_id=state.run_id),
            )

             # Attach debug trace to the returned answer (runtime-level diagnostics).
            if runtime_answer is not None:
                runtime_answer.trace_events = state.trace_events
                runtime_answer.run_id = run_id

            writer = self.context.trace_writer
            if writer is not None:
                metadata = RunMetadata(
                    run_id=state.run_id,
                    session_id=request.session_id,
                    user_id=request.user_id,
                    tenant_id=(request.tenant_id or self.context.config.tenant_id),
                    started_at_utc=state.started_at_utc,
                )
                writer.finalize_run(state.run_id, metadata)


    