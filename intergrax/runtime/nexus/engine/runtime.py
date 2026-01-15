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

import asyncio
import time
import uuid

from intergrax.llm_adapters.llm_usage_track import LLMUsageTracker
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetEnforcer, BudgetExceededError
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.errors.classifier import ErrorClassifier
from intergrax.runtime.nexus.messages.runtime_message_service import RuntimeMessageService
from intergrax.runtime.nexus.pipelines.contract import RuntimePipeline
from intergrax.runtime.nexus.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.nexus.policies.policy_enforcer import PolicyAbortError
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeRequest,
    RuntimeAnswer,
    StopReason,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.persistence_models import RunError, RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.runtime.runtime_run_abort import RuntimeRunAbortDiagV1
from intergrax.runtime.nexus.tracing.runtime.runtime_run_end import RuntimeRunEndDiagV1
from intergrax.runtime.nexus.tracing.runtime.runtime_run_retry import RuntimeRunRetryDiagV1
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
        self._message_service = RuntimeMessageService()


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, request: RuntimeRequest) -> RuntimeAnswer:
        """
        Main async entrypoint for the runtime.
        """

        run_id = f"run_{uuid.uuid4().hex}"
        start_perf = time.perf_counter()

        state = RuntimeState(
            context=self.context,
            request=request,
            run_id=run_id,
            llm_usage_tracker=LLMUsageTracker(run_id=run_id),
        )

        state.configure_llm_tracker()

        budget_enforcer: BudgetEnforcer | None = None
        if self.context.config.run_budget is not None and self.context.config.budget_policy is not None:
            budget_enforcer = BudgetEnforcer(
                budget=self.context.config.run_budget,
                policy=self.context.config.budget_policy,
            )

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
        run_error: RunError | None = None
        max_retries = int(state.context.config.max_run_retries)
        attempt = 0

        while True:
            try:
                
                runtime_answer = await self._run_with_timeout(pipeline=pipeline, state=state)       

                # --- Budget enforcement: max_llm_calls ---
                if state.llm_usage_tracker is not None:
                    report = state.llm_usage_tracker.build_report()
                    total_calls = report.total.calls

                    if budget_enforcer is not None and state.llm_usage_tracker is not None:
                        report = state.llm_usage_tracker.build_report()
                        total_calls = report.total.calls

                        budget_enforcer.check_llm_calls(
                            run_id=state.run_id,
                            llm_calls=total_calls,
                            state=state,
                        )

                        
                # --- Budget enforcement: max_tool_calls ---
                if budget_enforcer is not None:
                    budget_enforcer.check_tool_calls(
                        run_id=state.run_id,
                        tool_calls=len(state.tool_traces),
                        state=state,
                    )

                # --- Budget enforcement: max_total_tokens ---
                if budget_enforcer is not None and state.llm_usage_tracker is not None:
                    report = state.llm_usage_tracker.build_report()
                    total_tokens = report.total.total_tokens

                    budget_enforcer.check_total_tokens(
                        run_id=state.run_id,
                        total_tokens=total_tokens,
                        state=state,
                    )

                # --- Budget enforcement: max_wall_time_seconds ---
                if budget_enforcer is not None:
                    elapsed = time.perf_counter() - start_perf
                    budget_enforcer.check_wall_time(
                        run_id=state.run_id,
                        elapsed_seconds=elapsed,
                        state=state,
                    )

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
            
            except PolicyAbortError as exc:
                # Policy escalation (HITL) — not a system error, no retries.
                state.trace_event(
                    component=TraceComponent.POLICY,
                    step="hitl_escalation",
                    level=TraceLevel.WARNING,
                    message=str(exc),
                )

                message = (
                    state.context.config.hitl_default_message
                    or self._message_service.build_message(
                        stop_reason=StopReason.NEEDS_USER_INPUT,
                        state=state,
                        error=exc,
                    )
                )

                runtime_answer = RuntimeAnswer(
                    run_id=run_id,
                    answer=message,
                    stop_reason=StopReason.NEEDS_USER_INPUT,
                )

                return runtime_answer
            
            except BudgetExceededError as exc:
                # Budget exceeded is a controlled policy decision (same category as HITL),
                # not a system error and must not trigger retries.
                state.trace_event(
                    component=TraceComponent.POLICY,
                    step="hitl_escalation",
                    level=TraceLevel.WARNING,
                    message=str(exc),
                )

                message = (
                    state.context.config.hitl_default_message
                    or self._message_service.build_message(
                        stop_reason=StopReason.NEEDS_USER_INPUT,
                        state=state,
                        error=exc,
                    )
                )

                runtime_answer = RuntimeAnswer(
                    run_id=run_id,
                    answer=message,
                    stop_reason=StopReason.NEEDS_USER_INPUT,
                )

                return runtime_answer
            
            except Exception as ex:
                error_code = ErrorClassifier.classify(ex)

                retryable = error_code in state.context.config.retry_run_on
                if retryable and attempt < max_retries:
                    attempt += 1
                    state.trace_event(
                        component=TraceComponent.ENGINE,
                        step="run_retry",
                        level=TraceLevel.WARNING,
                        message="RuntimeEngine.run() retrying after failure.",
                        payload=RuntimeRunRetryDiagV1(
                            run_id=state.run_id,
                            attempt=attempt,
                            max_retries=max_retries,
                            error_code=error_code,
                        ),
                    )
                    continue

                # Final failure (no retries left or not retryable) -> persist error.
                run_error = RunError(
                    error_type=error_code,
                    message=str(ex),
                )
                raise

            finally:
                await state.finalize_llm_tracker(
                    request=request,
                    runtime_answer=runtime_answer,
                )

                if runtime_answer is None:
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

                duration_ms = int((time.perf_counter() - start_perf) * 1000)
                if duration_ms < 0:
                    duration_ms = 0
                
                llm_usage = state.llm_usage_tracker.export()

                writer = self.context.trace_writer

                if writer is not None:
                    metadata = RunMetadata(
                        run_id=state.run_id,
                        session_id=request.session_id,
                        user_id=request.user_id,
                        tenant_id=(request.tenant_id or self.context.config.tenant_id),
                        started_at_utc=state.started_at_utc,
                        stats=RunStats(
                            duration_ms=duration_ms,
                            llm_usage=llm_usage
                        ),
                        error=run_error,
                    )
                    writer.finalize_run(state.run_id, metadata)


    async def _run_with_timeout(
            self,
            *,
            pipeline: RuntimePipeline,
            state: RuntimeState,
    )->RuntimeAnswer:
        timeout_ms = state.context.config.runtime_timeout_ms
        if timeout_ms is None:
            return await pipeline.run(state=state)
        return await asyncio.wait_for(pipeline.run(state=state), timeout=timeout_ms/1000.0)