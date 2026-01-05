# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.engine.pipelines.pipeline_factory import PipelineFactory
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.engine_planner import EnginePlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor_models import PlanExecutionReport
from intergrax.runtime.drop_in_knowledge_mode.planning.step_planner import StepPlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import PlanBuildMode
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor import StepExecutor
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer

# Setup steps (outside planner)
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.session_and_ingest_step import SessionAndIngestStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.profile_based_memory_step import ProfileBasedMemoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.build_base_history_step import BuildBaseHistoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.history_step import HistoryStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.instructions_step import InstructionsStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.ensure_current_user_message_step import EnsureCurrentUserMessageStep


class PlannerStaticPipeline(RuntimePipeline):
    """
    STATIC plan pipeline:
      - deterministic setup outside planner (no LLM planning)
      - EnginePlanner -> StepPlanner(STATIC) -> StepExecutor(registry)
      - runtime_answer MUST be produced by FINALIZE_ANSWER handler (PersistAndBuildAnswerStep)
    """

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        # 1) Deterministic setup outside planner
        setup_steps = [
            SessionAndIngestStep(),
            ProfileBasedMemoryStep(),
            BuildBaseHistoryStep(),
            HistoryStep(),
            InstructionsStep(),
            EnsureCurrentUserMessageStep(),
        ]
        await self._execute_pipeline(setup_steps, state)

        # 2) Build components
        ctx = state.context
        cfg = ctx.config
        req = state.request

        if cfg.llm_adapter is None:
            raise RuntimeError("PlannerStaticPipeline: config.llm_adapter is required for EnginePlanner.")

        engine_planner = EnginePlanner(llm_adapter=cfg.llm_adapter)
        step_planner = StepPlanner(cfg.step_planner_cfg)

        registry = PipelineFactory.build_default_planning_step_registry()
        step_executor = StepExecutor(registry=registry, cfg=cfg.step_executor_cfg)

        # 3) Engine plan (LLM)
        engine_plan = await engine_planner.plan(
            req=req,
            state=state,
            config=cfg,
            prompt_config=cfg.planner_prompt_config,
            run_id=state.run_id,
        )

        # 4) Step plan (deterministic) + execute
        exec_plan = step_planner.build_from_engine_plan(
            user_message=req.message or "",
            engine_plan=engine_plan,
            plan_id=f"static-{state.run_id}",
            build_mode=PlanBuildMode.STATIC,
        )

        executed_plan: PlanExecutionReport = await step_executor.execute(plan=exec_plan, state=state)

        # 5) Enforce invariant: FINALIZE step must set runtime_answer
        if state.runtime_answer is None:
            raise RuntimeError("PlannerStaticPipeline: runtime_answer is not set. FINALIZE_ANSWER step failed or missing.")

        return state.runtime_answer
