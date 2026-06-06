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
from typing import Optional
import uuid

from intergrax.distributed.contracts.execution_semaphore import ExecutionSlot
from intergrax.llm_adapters.tracking.llm_usage_track import LLMUsageTracker
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
from intergrax.runtime.nexus.tracing.execution.execution_concurrency_diag import ExecutionConcurrencyDiagV1
from intergrax.runtime.nexus.tracing.persistence_models import RunError, RunMetadata, RunStats
from intergrax.runtime.nexus.tracing.runtime.runtime_run_abort import RuntimeRunAbortDiagV1
from intergrax.runtime.nexus.tracing.runtime.runtime_run_end import RuntimeRunEndDiagV1
from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
from intergrax.runtime.nexus.tracing.runtime.runtime_run_retry import RuntimeRunRetryDiagV1
from intergrax.runtime.nexus.tracing.runtime.harness_shadow_eval_recorded import (
    HarnessShadowEvalRecordedDiagV1,
)
from intergrax.runtime.nexus.tracing.runtime.runtime_run_start import RuntimeRunStartDiagV1
from intergrax.runtime.architecture.multi_agent_coordination import PlanningConstraints
from intergrax.runtime.adaptive.signal_emission import record_runtime_engine_outcome_signal
from intergrax.runtime.architecture.runtime_governance_bridge import RuntimeArchitectureGovernanceBridge
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


# ----------------------------------------------------------------------
# RuntimeEngine
# ----------------------------------------------------------------------


class RuntimeEngine:
    """
    High-level agent runtime engine for the Intergrax framework.

    Executes a configured :class:`~intergrax.runtime.nexus.pipelines.contract.RuntimePipeline`
    for a single agent run: session load, capability steps (RAG, websearch, tools),
    LLM synthesis, tracing, budget enforcement, and governance.

    Invoked by :class:`~intergrax.agents.agent_engine.AgentEngine` and
    :class:`~intergrax.runtime.nexus.nexus_loop.NexusLoop` — not called directly by applications.
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

        if not request.tenant_id:
            raise ValueError("tenant_id must be provided in RuntimeRequest.")

        run_id = request.metadata.get("run_id") or f"run_{uuid.uuid4().hex}"
        start_perf = time.perf_counter()

        state = RuntimeState(
            context=self.context,
            request=request,
            run_id=run_id,
            llm_usage_tracker=LLMUsageTracker(run_id=run_id),
        )

        state.configure_llm_tracker()

        slot: Optional[ExecutionSlot] = None
        acquire_perf: float | None = None

        if self.context.execution_semaphore is not None:
            if self.context.max_parallel_per_tenant is None:
                raise RuntimeError(
                    "execution_semaphore configured but max_parallel_per_tenant not set"
                )

            slot = self.context.execution_semaphore.acquire(
                tenant_id=state.tenant_id,
                max_parallel=self.context.max_parallel_per_tenant,
            )

            if slot is None:
                state.trace_event(
                    component=TraceComponent.ENGINE,
                    step="execution.acquire.rejected",
                    level=TraceLevel.WARNING,
                    message="Execution concurrency limit reached.",
                    payload=ExecutionConcurrencyDiagV1(
                        tenant_id=state.tenant_id,
                        run_id=state.run_id,
                        action="acquire_rejected",
                    ),
                )

                raise RuntimeError(
                    f"Execution concurrency limit reached for tenant {state.tenant_id}"
                )
            
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="execution.acquire.success",
                level=TraceLevel.INFO,
                message="Execution slot acquired.",
                payload=ExecutionConcurrencyDiagV1(
                    tenant_id=state.tenant_id,
                    run_id=state.run_id,
                    action="acquire_success",
                ),
            )

            acquire_perf = time.perf_counter()

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
                tenant_id=request.tenant_id,
                run_id=state.run_id,
                pipeline_name=pipeline.__class__.__name__ if pipeline is not None else "None",
            ),
        )
        planning_constraints_raw = request.metadata.get("planning_constraints")
        if isinstance(planning_constraints_raw, dict):
            governance_bridge = RuntimeArchitectureGovernanceBridge()
            planning_constraints = PlanningConstraints.model_validate(planning_constraints_raw)
            governance_metadata = governance_bridge.build_trace_metadata(
                constraints=planning_constraints
            )
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="architecture_governance",
                level=TraceLevel.INFO,
                message="Architecture governance metadata recorded.",
                payload=governance_metadata,
            )

        runtime_answer: RuntimeAnswer | None = None
        run_error: RunError | None = None
        max_retries = int(state.context.config.max_run_retries)
        attempt = 0

        try:
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

                    # --- Hard Output Gate (contract enforcement) ---
                    self._validate_runtime_answer_contract(
                        state=state,
                        runtime_answer=runtime_answer,
                    )

                    self._maybe_record_harness_shadow_evaluation(
                        request=request,
                        state=state,
                        runtime_answer=runtime_answer,
                    )
                    self._maybe_record_adaptive_outcome_signal(
                        request=request,
                        state=state,
                        runtime_answer=runtime_answer,
                        start_perf=start_perf,
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

                    coordinator = RetryCoordinator(
                        max_run_retries=max_retries,
                        retry_run_on=state.context.config.retry_run_on,
                    )
                    if coordinator.should_retry_run(
                        attempt=attempt,
                        error_code=error_code,
                    ):
                        attempt += 1
                        state.trace_event(
                            component=TraceComponent.ENGINE,
                            step="retry_scheduled",
                            level=TraceLevel.WARNING,
                            message="RuntimeEngine.run() retry scheduled after failure.",
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
                            tenant_id=request.tenant_id,
                            started_at_utc=state.started_at_utc,
                            stats=RunStats(
                                duration_ms=duration_ms,
                                llm_usage=llm_usage
                            ),
                            error=run_error,
                        )
                        writer.finalize_run(state.run_id, metadata)

                    governance = self.context.governance_service
                    if governance is not None:
                        
                        try:
                            governance.evaluate(run_id=state.run_id, agent_id=request.agent_id)
                            state.trace_event(
                                component=TraceComponent.POLICY,
                                step="governance_evaluated",
                                level=TraceLevel.INFO,
                                message="Governance evaluation completed.",
                            )
                        except Exception as exc:
                            state.trace_event(
                                component=TraceComponent.POLICY,
                                step="governance_failed",
                                level=TraceLevel.WARNING,
                                message=f"Governance evaluation failed: {exc.__class__.__name__}: {exc}",
                                )

        finally:
            if slot is not None:
                duration_ms = None
                if acquire_perf is not None:
                    duration_ms = int((time.perf_counter() - acquire_perf) * 1000)
                    if duration_ms < 0:
                        duration_ms = 0

                self.context.execution_semaphore.release(
                    tenant_id=state.tenant_id,
                    slot=slot,
                )

                state.trace_event(
                    component=TraceComponent.ENGINE,
                    step="execution.release",
                    level=TraceLevel.INFO,
                    message="Execution slot released.",
                    payload=ExecutionConcurrencyDiagV1(
                        tenant_id=state.tenant_id,
                        run_id=state.run_id,
                        action="release",
                        duration_ms=duration_ms,
                    ),
                )

                threshold = self.context.config.execution_slot_warn_threshold_ms
                if (
                    threshold is not None
                    and duration_ms is not None
                    and duration_ms >= threshold
                ):
                    state.trace_event(
                        component=TraceComponent.ENGINE,
                        step="execution.slot_long_hold_warning",
                        level=TraceLevel.WARNING,
                        message="Execution slot held longer than configured threshold.",
                        payload=ExecutionConcurrencyDiagV1(
                            tenant_id=state.tenant_id,
                            run_id=state.run_id,
                            action="release",
                            duration_ms=duration_ms,
                        ),
                    )

    def _maybe_record_harness_shadow_evaluation(
        self,
        *,
        request: RuntimeRequest,
        state: RuntimeState,
        runtime_answer: RuntimeAnswer,
    ) -> None:
        """Optional shadow evaluation when ``request.metadata`` requests it (W-OPS.11)."""
        profile = state.context.config.evaluation_profile
        if profile is not None and not profile.shadow_eval_enabled:
            return
        raw = request.metadata.get("harness_shadow_eval")
        if not isinstance(raw, dict):
            return
        scenario_id = str(raw.get("scenario_id") or "harness.default")
        passed = bool(raw.get("passed", runtime_answer.answer.strip() != ""))
        score_raw = raw.get("score", 1.0 if passed else 0.0)
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 1.0 if passed else 0.0
        bridge = RuntimeArchitectureGovernanceBridge(
            evaluation_registry=state.context.config.evaluation_registry,
        )
        observation = bridge.record_shadow_run_evaluation(
            run_id=state.run_id,
            agent_id=request.agent_id,
            scenario_id=scenario_id,
            passed=passed,
            score=score,
        )
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="harness_shadow_eval_recorded",
            level=TraceLevel.INFO,
            message="Harness shadow evaluation observation recorded.",
            payload=HarnessShadowEvalRecordedDiagV1(
                run_id=observation.run_id,
                agent_id=observation.agent_id,
                scenario_id=observation.scenario_id,
                passed=observation.passed,
                score=observation.score,
                observation_id=observation.observation_id,
            ),
        )

    def _maybe_record_adaptive_outcome_signal(
        self,
        *,
        request: RuntimeRequest,
        state: RuntimeState,
        runtime_answer: RuntimeAnswer,
        start_perf: float,
    ) -> None:
        """Optional adaptive signal when ``RuntimeConfig.signal_collector`` is wired (W-ADAPT-1.11)."""
        collector = state.context.config.signal_collector
        if collector is None:
            return
        elapsed_ms = int((time.perf_counter() - start_perf) * 1000)
        total_tokens = 0
        actual_cost: float | None = None
        if state.llm_usage_tracker is not None:
            report = state.llm_usage_tracker.build_report()
            total_tokens = report.total.total_tokens
            actual_cost = float(report.total.cost) if report.total.cost is not None else None
        record_runtime_engine_outcome_signal(
            collector,
            request=request,
            run_id=state.run_id,
            answer=runtime_answer.answer,
            latency_ms=elapsed_ms,
            total_tokens=total_tokens,
            actual_cost=actual_cost,
            run_budget=state.context.config.run_budget,
            evaluation_registry=state.context.config.evaluation_registry,
        )

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
    

    
    def _validate_runtime_answer_contract(
        self,
        *,
        state: RuntimeState,
        runtime_answer: RuntimeAnswer,
    ) -> None:
        """
        Hard production contract validation for RuntimeAnswer.

        This enforces ENGINE-level invariants.
        It does NOT perform semantic or content inspection.
        """

        if runtime_answer is None:
            from intergrax.runtime.nexus.errors.output_validation_error import OutputValidationError
            raise OutputValidationError(
                run_id=state.run_id,
                reason_code="NULL_RUNTIME_ANSWER",
                message="RuntimeAnswer is None.",
            )

        if not isinstance(runtime_answer.answer, str):
            from intergrax.runtime.nexus.errors.output_validation_error import OutputValidationError
            raise OutputValidationError(
                run_id=state.run_id,
                reason_code="INVALID_ANSWER_TYPE",
                message="RuntimeAnswer.answer must be a string.",
            )

        if not runtime_answer.answer.strip():
            from intergrax.runtime.nexus.errors.output_validation_error import OutputValidationError
            raise OutputValidationError(
                run_id=state.run_id,
                reason_code="EMPTY_OUTPUT",
                message="RuntimeAnswer.answer is empty.",
            )

