# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan
from intergrax.runtime.nexus.planning.plan_loop_models import OnMaxReplansPolicy, PlanLoopPolicy
from intergrax.runtime.nexus.planning.step_executor_models import (
    PlanExecutionReport,
    PlanStopReason,
    ReplanContext,
    ReplanFailedStep,
    StepStatus,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeAnswer
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.stepplan_models import PlanBuildMode
from intergrax.runtime.nexus.planning.engine_planner import EnginePlanner
from intergrax.runtime.nexus.planning.step_planner import StepPlanner
from intergrax.runtime.nexus.planning.step_executor import StepExecutor
from intergrax.runtime.nexus.tracing.plan.dynamic_engine_plan_produced import PlannerDynamicEnginePlanProducedDiagV1
from intergrax.runtime.nexus.tracing.plan.execution_requested_replan import PlannerExecutionRequestedReplanDiagV1
from intergrax.runtime.nexus.tracing.plan.iteration_completed_continue import PlannerIterationCompletedContinueDiagV1
from intergrax.runtime.nexus.tracing.plan.planner_build_debug import PlannerBuildDebugDiagV1
from intergrax.runtime.nexus.tracing.plan.planning_iteration_started import PlannerPlanningIterationStartedDiagV1
from intergrax.runtime.nexus.tracing.plan.static_engine_plan_produced import PlannerStaticEnginePlanProducedDiagV1
from intergrax.runtime.nexus.tracing.plan.static_execution_requested_replan import PlannerStaticExecutionRequestedReplanDiagV1
from intergrax.runtime.nexus.tracing.plan.static_planning_iteration_started import PlannerStaticPlanningIterationStartedDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel



class PlanLoopController:
    """
    Production-grade controller for plan->execute->interpret with bounded replanning.

    This is the single source of truth for:
    - replanning loop limits
    - stop_reason interpretation (HITL/REPLAN/FAILED/COMPLETED)
    - escalation strategy when stuck
    """

    def __init__(
        self,
        *,
        engine_planner: EnginePlanner,
        step_planner: StepPlanner,
        step_executor: StepExecutor,
        policy: Optional[PlanLoopPolicy] = None,
    ) -> None:
        self._engine_planner = engine_planner
        self._step_planner = step_planner
        self._step_executor = step_executor
        self._policy = policy or PlanLoopPolicy()

    
    def _mk_iteration_ctx(
        self,
        *,
        attempt: int,
        last_plan_id: str,
        reason: str,
        report: PlanExecutionReport,
    ) -> ReplanContext:
        """
        DYNAMIC loop needs a stable attempt counter even for non-error iterations
        (e.g. WEBSEARCH -> SYNTHESIZE -> FINALIZE). This is also required for
        deterministic ScriptedPlanSource selection by req.replan_ctx.attempt.

        We keep it low-volume and safe: no large outputs, only executed order + failures.
        """
        executed_order = [
            sid.value if hasattr(sid, "value") else str(sid)
            for sid in (report.executed_order or [])
        ]

        failed_steps = []
        skipped_with_error_steps = []

        for r in report.step_results.values():
            if r.status == StepStatus.FAILED:
                failed_steps.append(
                    ReplanFailedStep(
                        step_id=r.step_id.value,
                        action=r.action.value,
                        error_code=(r.error.code.value if r.error is not None else None),
                        error_message=(r.error.message if r.error is not None else None),
                    )
                )
            elif r.status == StepStatus.SKIPPED_WITH_ERROR:
                skipped_with_error_steps.append(
                    ReplanFailedStep(
                        step_id=r.step_id.value,
                        action=r.action.value,
                        error_code=(r.error.code.value if r.error is not None else None),
                        error_message=(r.error.message if r.error is not None else None),
                    )
                )

        return ReplanContext(
            attempt=int(attempt),
            last_plan_id=str(last_plan_id),
            replan_reason=str(reason),
            executed_order=executed_order,
            failed_steps=failed_steps,
            skipped_with_error_steps=skipped_with_error_steps,
        )

    async def run_dynamic(
        self,
        *,
        state: RuntimeState,
        plan_id_prefix: str,
        user_message: str,
    ) -> RuntimeAnswer:
        """
        DYNAMIC mode:
          - EnginePlanner produces an EnginePlan with next_step (single-step intent)
          - StepPlanner builds a ONE-STEP ExecutionPlan (PlanBuildMode.DYNAMIC)
          - StepExecutor runs it
          - Loop continues until:
              - FINAL answer is produced (state.runtime_answer set), or
              - NEEDS_USER_INPUT (HITL), or
              - FAILED, or
              - limits are exceeded (max_replans treated as max iterations budget)
        """
        iterations_used = 0
        same_plan_repeats = 0
        last_engine_plan_fingerprint: Optional[str] = None

        replan_ctx: Optional[ReplanContext] = None

        while True:
            cfg = state.context.config
            req = state.request

            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan_loop_dynamic",
                message="Planning iteration started.",
                level=TraceLevel.INFO,
                payload=PlannerPlanningIterationStartedDiagV1(
                    iterations_used=iterations_used,
                    same_plan_repeats=same_plan_repeats,
                    has_replan_ctx=replan_ctx is not None,
                    replan_reason=(replan_ctx.replan_reason or "").strip() if replan_ctx is not None else None,
                    replan_attempt=(replan_ctx.attempt if replan_ctx is not None else None),
                ),
            )

            engine_plan: EnginePlan = await self._engine_planner.plan(
                req=req,
                state=state,
                config=cfg,
                prompt_config=cfg.planner_prompt_config,
                run_id=state.run_id,
                replan_ctx=replan_ctx,
            )

            current_fp = engine_plan.fingerprint()

            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan_loop_dynamic",
                message="Engine plan produced.",
                level=TraceLevel.INFO,
                payload=PlannerDynamicEnginePlanProducedDiagV1(
                    fingerprint=current_fp,
                    intent=engine_plan.intent.value,
                    next_step=engine_plan.next_step.value if engine_plan.next_step is not None else None,
                    ask_clarifying_question=engine_plan.ask_clarifying_question,
                    use_websearch=engine_plan.use_websearch,
                    use_user_longterm_memory=engine_plan.use_user_longterm_memory,
                    use_rag=engine_plan.use_rag,
                    use_tools=engine_plan.use_tools,
                    same_plan_repeats=same_plan_repeats,
                ),
            )

            if last_engine_plan_fingerprint == current_fp:
                same_plan_repeats += 1
            else:
                same_plan_repeats = 0
            last_engine_plan_fingerprint = current_fp

            if same_plan_repeats >= self._policy.max_same_plan_repeats:
                reason = (replan_ctx.replan_reason or "").strip() if replan_ctx is not None else None
                msg = self._policy.hitl_escalation_message_builder(reason)

                ans = RuntimeAnswer(answer=msg)
                ans.route.strategy = "hitl_escalation_same_plan_repeat"
                ans.route.extra["stop_reason"] = "NEEDS_USER_INPUT"
                ans.route.extra["iterations_used"] = iterations_used
                ans.route.extra["same_plan_repeats"] = same_plan_repeats
                ans.route.extra["same_plan_repeat_limit"] = self._policy.max_same_plan_repeats
                ans.route.extra["last_fingerprint"] = last_engine_plan_fingerprint
                state.runtime_answer = ans
                return ans

            exec_plan = self._step_planner.build_from_engine_plan(
                user_message=(user_message or ""),
                engine_plan=engine_plan,
                plan_id=f"{plan_id_prefix}-i{iterations_used}",
                build_mode=PlanBuildMode.DYNAMIC,
            )

            report: PlanExecutionReport = await self._step_executor.execute(plan=exec_plan, state=state)

            # ----------------------------
            # Interpret stop_reason
            # ----------------------------
            if report.stop_reason == PlanStopReason.NEEDS_USER_INPUT:
                if report.user_input_request is None:
                    raise RuntimeError(
                        "PlanLoopController(DYNAMIC): stop_reason=NEEDS_USER_INPUT but user_input_request is None."
                    )

                ans = RuntimeAnswer(answer=report.user_input_request.question)
                ans.route.strategy = "hitl_clarify"
                ans.route.extra["stop_reason"] = report.stop_reason.value
                ans.route.extra["origin_step_id"] = report.user_input_request.origin_step_id
                ans.route.extra["context_key"] = report.user_input_request.context_key
                ans.route.extra["must_answer_to_continue"] = report.user_input_request.must_answer_to_continue

                state.runtime_answer = ans
                return ans

            if report.stop_reason == PlanStopReason.FAILED:
                last_failed = None
                for r in report.step_results.values():
                    if r.status == StepStatus.FAILED:
                        last_failed = r

                if last_failed is not None and last_failed.error is not None:
                    raise RuntimeError(
                        "PlanLoopController(DYNAMIC): plan execution FAILED. "
                        f"step_id={last_failed.step_id.value} action={last_failed.action.value} "
                        f"error={last_failed.error.code.value}: {last_failed.error.message}"
                    )
                raise RuntimeError("PlanLoopController(DYNAMIC): plan execution FAILED.")

            if report.stop_reason == PlanStopReason.REPLAN_REQUIRED:
                iterations_used += 1
                if iterations_used > self._policy.max_replans:
                    if self._policy.on_max_replans is OnMaxReplansPolicy.HITL:
                        reason = (report.replan_reason or "").strip() or None
                        msg = self._policy.hitl_escalation_message_builder(reason)

                        ans = RuntimeAnswer(answer=msg)
                        ans.route.strategy = "hitl_escalation_max_replans"
                        ans.route.extra["stop_reason"] = "NEEDS_USER_INPUT"
                        ans.route.extra["iterations_used"] = iterations_used
                        ans.route.extra["max_replans"] = self._policy.max_replans
                        ans.route.extra["replan_reason"] = report.replan_reason
                        state.runtime_answer = ans
                        return ans

                    raise RuntimeError(
                        "PlanLoopController(DYNAMIC): max replans exceeded. "
                        f"max_replans={self._policy.max_replans} iterations_used={iterations_used} "
                        f"replan_reason={report.replan_reason!r}"
                    )

                replan_ctx = ReplanContext.from_report(
                    report=report,
                    attempt=iterations_used,
                    last_plan_id=exec_plan.plan_id,
                )

                state.runtime_answer = None

                state.trace_event(
                    component=TraceComponent.PLANNER,
                    step="plan_loop_dynamic",
                    message="Execution requested replan.",
                    level=TraceLevel.INFO,
                    payload=PlannerExecutionRequestedReplanDiagV1(
                        iterations_used=iterations_used,
                        replan_reason=report.replan_reason,
                        last_plan_id=exec_plan.plan_id,
                    ),
                )

                continue

            # COMPLETED
            # If FINALIZE_ANSWER happened, runtime_answer MUST be set.
            ans = state.runtime_answer
            if ans is not None:
                return ans

            # Otherwise: completed a context-building step (websearch/rag/tools/draft, etc.),
            # so we iterate again with a synthetic iteration ctx to bump attempt.
            iterations_used += 1
            if iterations_used > self._policy.max_replans:
                if self._policy.on_max_replans is OnMaxReplansPolicy.HITL:
                    msg = self._policy.hitl_escalation_message_builder("max_iterations_exceeded")

                    ans = RuntimeAnswer(answer=msg)
                    ans.route.strategy = "hitl_escalation_max_iterations"
                    ans.route.extra["stop_reason"] = "NEEDS_USER_INPUT"
                    ans.route.extra["iterations_used"] = iterations_used
                    ans.route.extra["max_replans"] = self._policy.max_replans
                    ans.route.extra["last_plan_id"] = exec_plan.plan_id
                    state.runtime_answer = ans
                    return ans

                raise RuntimeError(
                    "PlanLoopController(DYNAMIC): max iterations exceeded. "
                    f"max_replans={self._policy.max_replans} iterations_used={iterations_used}"
                )

            replan_ctx = self._mk_iteration_ctx(
                attempt=iterations_used,
                last_plan_id=exec_plan.plan_id,
                reason="iteration_completed",
                report=report,
            )

            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan_loop_dynamic",
                message="Iteration completed; continuing to next dynamic step.",
                level=TraceLevel.INFO,
                payload=PlannerIterationCompletedContinueDiagV1(
                    iterations_used=iterations_used,
                    last_plan_id=exec_plan.plan_id,
                    replan_attempt=replan_ctx.attempt,
                ),
            )

            continue


    async def run_static(
        self,
        *,
        state: RuntimeState,
        plan_id_prefix: str,
        user_message: str,
    ) -> RuntimeAnswer:
        """
        STATIC mode: StepPlanner builds a full EXECUTE plan; executor runs it.
        Supports bounded replanning with structured feedback passed to EnginePlanner.
        """
        replans_used = 0
        same_plan_repeats = 0
        last_engine_plan_fingerprint: Optional[str] = None

        replan_ctx: Optional[ReplanContext] = None

        while True:
            cfg = state.context.config
            req = state.request

            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan_loop_static",
                message="Planning iteration started.",
                level=TraceLevel.INFO,
                payload=PlannerStaticPlanningIterationStartedDiagV1(
                    replans_used=replans_used,
                    same_plan_repeats=same_plan_repeats,
                    has_replan_ctx=replan_ctx is not None,
                    replan_reason=(replan_ctx.replan_reason or "").strip() if replan_ctx is not None else None,
                ),
            )

            engine_plan: EnginePlan = await self._engine_planner.plan(
                req=req,
                state=state,
                config=cfg,
                prompt_config=cfg.planner_prompt_config,
                run_id=state.run_id,
                replan_ctx=replan_ctx,
            )

            # Fingerprint must be stable.
            current_fp = engine_plan.fingerprint()

            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan_loop_static",
                message="Engine plan produced.",
                level=TraceLevel.INFO,
                payload=PlannerStaticEnginePlanProducedDiagV1(
                    fingerprint=current_fp,
                    intent=engine_plan.intent.value,
                    next_step=engine_plan.next_step.value if engine_plan.next_step is not None else None,
                    ask_clarifying_question=engine_plan.ask_clarifying_question,
                    use_websearch=engine_plan.use_websearch,
                    use_user_longterm_memory=engine_plan.use_user_longterm_memory,
                    use_rag=engine_plan.use_rag,
                    use_tools=engine_plan.use_tools,
                    same_plan_repeats=same_plan_repeats,
                ),
            )

            if last_engine_plan_fingerprint == current_fp:
                same_plan_repeats += 1
            else:
                same_plan_repeats = 0
            last_engine_plan_fingerprint = current_fp

            if same_plan_repeats >= self._policy.max_same_plan_repeats:
                reason = None
                if replan_ctx is not None:
                    reason = (replan_ctx.replan_reason or "").strip() or None

                msg = self._policy.hitl_escalation_message_builder(reason)

                ans = RuntimeAnswer(answer=msg)
                ans.route.strategy = "hitl_escalation_same_plan_repeat"
                ans.route.extra["stop_reason"] = "NEEDS_USER_INPUT"
                ans.route.extra["replans_used"] = replans_used
                ans.route.extra["same_plan_repeats"] = same_plan_repeats
                ans.route.extra["same_plan_repeat_limit"] = self._policy.max_same_plan_repeats
                ans.route.extra["last_fingerprint"] = last_engine_plan_fingerprint
                state.runtime_answer = ans
                return ans

            exec_plan = self._step_planner.build_from_engine_plan(
                user_message=user_message or "",
                engine_plan=engine_plan,
                plan_id=f"{plan_id_prefix}-r{replans_used}",
                build_mode=PlanBuildMode.STATIC,
            )

            report: PlanExecutionReport = await self._step_executor.execute(plan=exec_plan, state=state)

            # ----------------------------
            # Interpret stop_reason
            # ----------------------------
            if report.stop_reason == PlanStopReason.NEEDS_USER_INPUT:
                if report.user_input_request is None:
                    raise RuntimeError(
                        "PlanLoopController(STATIC): stop_reason=NEEDS_USER_INPUT but user_input_request is None."
                    )

                ans = RuntimeAnswer(answer=report.user_input_request.question)
                ans.route.strategy = "hitl_clarify"
                ans.route.extra["stop_reason"] = report.stop_reason.value
                ans.route.extra["origin_step_id"] = report.user_input_request.origin_step_id
                ans.route.extra["context_key"] = report.user_input_request.context_key
                ans.route.extra["must_answer_to_continue"] = report.user_input_request.must_answer_to_continue

                state.runtime_answer = ans
                return ans

            if report.stop_reason == PlanStopReason.REPLAN_REQUIRED:
                replans_used += 1
                if replans_used > self._policy.max_replans:
                    if self._policy.on_max_replans is OnMaxReplansPolicy.HITL:
                        reason = (report.replan_reason or "").strip() or None
                        msg = self._policy.hitl_escalation_message_builder(reason)

                        ans = RuntimeAnswer(answer=msg)
                        ans.route.strategy = "hitl_escalation_max_replans"
                        ans.route.extra["stop_reason"] = "NEEDS_USER_INPUT"
                        ans.route.extra["replans_used"] = replans_used
                        ans.route.extra["max_replans"] = self._policy.max_replans
                        ans.route.extra["replan_reason"] = report.replan_reason
                        state.runtime_answer = ans
                        return ans

                    raise RuntimeError(
                        "PlanLoopController(STATIC): max replans exceeded. "
                        f"max_replans={self._policy.max_replans} replans_used={replans_used} "
                        f"replan_reason={report.replan_reason!r}"
                    )

                replan_ctx = ReplanContext.from_report(
                    report=report,
                    attempt=replans_used,
                    last_plan_id=exec_plan.plan_id,
                )

                state.runtime_answer = None

                state.trace_event(
                    component=TraceComponent.PLANNER,
                    step="plan_loop_static",
                    message="Execution requested replan.",
                    level=TraceLevel.INFO,
                    payload=PlannerStaticExecutionRequestedReplanDiagV1(
                        replans_used=replans_used,
                        replan_reason=report.replan_reason,
                        last_plan_id=exec_plan.plan_id,
                    ),
                )

                continue

            if report.stop_reason == PlanStopReason.FAILED:
                last_failed = None
                for r in report.step_results.values():
                    if r.status == StepStatus.FAILED:
                        last_failed = r

                if last_failed is not None and last_failed.error is not None:
                    raise RuntimeError(
                        "PlanLoopController(STATIC): plan execution FAILED. "
                        f"step_id={last_failed.step_id.value} action={last_failed.action.value} "
                        f"error={last_failed.error.code.value}: {last_failed.error.message}"
                    )
                raise RuntimeError("PlanLoopController(STATIC): plan execution FAILED.")

            # COMPLETED: invariant should hold
            ans = state.runtime_answer
            if ans is None:
                raise RuntimeError(
                    "PlanLoopController(STATIC): state.runtime_answer is not set after COMPLETED. "
                    "FINALIZE_ANSWER step failed/missing."
                )
            return ans

    