# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Optional


from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.drop_in_knowledge_mode.planning.engine_planner import EnginePlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.step_planner import StepPlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor import StepExecutor
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import PlanBuildMode
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import PlanExecutionReport, PlanStopReason, StepStatus

from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.build_base_history_step import BuildBaseHistoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.history_step import HistoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.instructions_step import InstructionsStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.profile_based_memory_step import ProfileBasedMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.session_and_ingest_step import SessionAndIngestStep


class PlannerStaticPipeline(RuntimePipeline):
    """
    STATIC plan pipeline:
      - deterministic setup outside planner
      - EnginePlanner -> StepPlanner(STATIC) -> StepExecutor(registry)
      - runtime_answer MUST be produced by PersistAndBuildAnswerStep (or HITL wrapper in pipeline)
    """

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        # ---------------------------------------------------------------------
        # 0) Validate configuration early (fail fast, before planning/execution)
        # ---------------------------------------------------------------------
        ctx = state.context
        cfg = ctx.config
        req = state.request

        if cfg.llm_adapter is None:
            raise RuntimeError("PlannerStaticPipeline: config.llm_adapter is required for EnginePlanner.")
        if cfg.step_planner_cfg is None:
            raise RuntimeError("PlannerStaticPipeline: config.step_planner_cfg is required.")
        if cfg.step_executor_cfg is None:
            raise RuntimeError("PlannerStaticPipeline: config.step_executor_cfg is required.")
        if cfg.planner_prompt_config is None:
            raise RuntimeError("PlannerStaticPipeline: config.planner_prompt_config is required.")

        # ---------------------------------------------------------------------
        # 1) Deterministic setup outside planner (same category as NoPlannerPipeline setup)
        # ---------------------------------------------------------------------
        setup_steps = [
            SessionAndIngestStep(),
            ProfileBasedMemoryStep(),
            BuildBaseHistoryStep(),
            HistoryStep(),
            InstructionsStep(),
            EnsureCurrentUserMessageStep(),
        ]
        await self._execute_pipeline(setup_steps, state)

        # ---------------------------------------------------------------------
        # 2) Build components from config/state (no new protocols/classes)
        # ---------------------------------------------------------------------
        engine_planner = EnginePlanner(llm_adapter=cfg.llm_adapter)
        step_planner = StepPlanner(cfg.step_planner_cfg)

        registry = PipelineFactory.build_default_planning_step_registry()
        step_executor = StepExecutor(registry=registry, cfg=cfg.step_executor_cfg)

        # ---------------------------------------------------------------------
        # 3-4) Plan -> Execute with minimal replanning loop (STATIC)
        # ---------------------------------------------------------------------
        max_replans = 1
        replans_used = 0

        while True:
            engine_plan = await engine_planner.plan(
                req=req,
                state=state,
                config=cfg,
                prompt_config=cfg.planner_prompt_config,
                run_id=state.run_id,
            )

            exec_plan = step_planner.build_from_engine_plan(
                user_message=req.message or "",
                engine_plan=engine_plan,
                plan_id=f"static-{state.run_id}-r{replans_used}",
                build_mode=PlanBuildMode.STATIC,
            )

            report: PlanExecutionReport = await step_executor.execute(plan=exec_plan, state=state)
            last_report = report

            # -----------------------------------------------------------------
            # Interpret stop_reason (STATIC contract)
            # -----------------------------------------------------------------
            if report.stop_reason == PlanStopReason.NEEDS_USER_INPUT:
                if report.user_input_request is None:
                    raise RuntimeError(
                        "PlannerStaticPipeline: stop_reason=NEEDS_USER_INPUT but user_input_request is None."
                    )

                runtime_answer = RuntimeAnswer(answer=report.user_input_request.question)

                runtime_answer.route.strategy = "hitl_clarify"
                runtime_answer.route.extra["stop_reason"] = report.stop_reason.value
                runtime_answer.route.extra["origin_step_id"] = report.user_input_request.origin_step_id
                runtime_answer.route.extra["context_key"] = report.user_input_request.context_key
                runtime_answer.route.extra["must_answer_to_continue"] = (
                    report.user_input_request.must_answer_to_continue
                )

                state.runtime_answer = runtime_answer
                return runtime_answer

            if report.stop_reason == PlanStopReason.REPLAN_REQUIRED:
                replans_used += 1
                if replans_used > max_replans:
                    raise RuntimeError(
                        "PlannerStaticPipeline: plan execution requested REPLAN but max replans exceeded. "
                        f"max_replans={max_replans} replans_used={replans_used} "
                        f"replan_reason={report.replan_reason!r}"
                    )
                # Continue loop: ask EnginePlanner again and rebuild plan.
                continue

            if report.stop_reason == PlanStopReason.FAILED:
                # Best-effort diagnostics: show last failed step if available.
                last_failed = None
                for r in report.step_results.values():
                    if r.status == StepStatus.FAILED:
                        last_failed = r
                if last_failed is not None and last_failed.error is not None:
                    raise RuntimeError(
                        "PlannerStaticPipeline: plan execution FAILED. "
                        f"step_id={last_failed.step_id.value} action={last_failed.action.value} "
                        f"error={last_failed.error.code.value}: {last_failed.error.message}"
                    )
                raise RuntimeError("PlannerStaticPipeline: plan execution FAILED.")

            # COMPLETED: enforce invariant
            runtime_answer = state.runtime_answer
            if runtime_answer is None:
                raise RuntimeError(
                    "PlannerStaticPipeline: state.runtime_answer is not set. "
                    "FINALIZE_ANSWER step failed/missing or pipeline didn't translate stop_reason yet."
                )

            return runtime_answer


