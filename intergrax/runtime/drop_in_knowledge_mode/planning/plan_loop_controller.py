# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from intergrax.runtime.drop_in_knowledge_mode.planning.engine_plan_models import EnginePlan
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import (
    PlanExecutionReport,
    PlanStopReason,
    ReplanContext,
    StepStatus,
)
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import PlanBuildMode
from intergrax.runtime.drop_in_knowledge_mode.planning.engine_planner import EnginePlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.step_planner import StepPlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor import StepExecutor


@dataclass(frozen=True)
class PlanLoopPolicy:
    max_replans: int = 1
    # If replanning produces the same engine plan repeatedly, fail fast / escalate.
    max_same_plan_repeats: int = 1

    on_max_replans: str = "raise"  # "raise" | "hitl"

    # Builder for escalation HITL message
    hitl_escalation_message_builder: Callable[[Optional[str]], str] = (
        lambda reason: (
            "I need one clarification to continue."
            + (f" Replan reason: {reason}" if reason else "")
            + " Please clarify what you want me to do next."
        )
    )


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
                component="planner",
                step="plan_loop_static",
                message="Planning iteration started.",
                data={
                    "replans_used": replans_used,
                    "same_plan_repeats": same_plan_repeats,
                    "has_replan_ctx": replan_ctx is not None,
                    "replan_reason": (replan_ctx.replan_reason or "").strip() if replan_ctx is not None else None,
                },
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

            # Extract planner debug info (may be None)
            plan_dbg = engine_plan.debug or {}

            # Production-grade trace: make forced_plan usage observable in tests/logs
            state.trace_event(
                component="planner",
                step="plan_loop_static",
                message="Engine plan produced.",
                data={
                    "fingerprint": current_fp,
                    "intent": engine_plan.intent.value,
                    "next_step": engine_plan.next_step.value if engine_plan.next_step is not None else None,
                    "ask_clarifying_question": engine_plan.ask_clarifying_question,
                    "use_websearch": engine_plan.use_websearch,
                    "use_user_longterm_memory": engine_plan.use_user_longterm_memory,
                    "use_rag": engine_plan.use_rag,
                    "use_tools": engine_plan.use_tools,
                    "same_plan_repeats": same_plan_repeats,
                    "capability_clamp": plan_dbg.get("capability_clamp"),
                    # New: deterministic override / replan injection visibility
                    "planner_forced_plan_used": plan_dbg.get("planner_forced_plan_used"),
                    "planner_forced_plan_hash": plan_dbg.get("planner_forced_plan_hash"),
                    "planner_replan_ctx_present": plan_dbg.get("planner_replan_ctx_present"),
                    "planner_replan_ctx_hash": plan_dbg.get("planner_replan_ctx_hash"),
                },
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
                    if self._policy.on_max_replans == "hitl":
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
                    component="planner",
                    step="plan_loop_static",
                    message="Execution requested replan.",
                    data={
                        "replans_used": replans_used,
                        "replan_reason": report.replan_reason,
                        "last_plan_id": exec_plan.plan_id,
                    },
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

