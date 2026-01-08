# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.pipelines.contract import RuntimePipeline
from intergrax.runtime.drop_in_knowledge_mode.planning.engine_planner import EnginePlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.plan_loop_controller import PlanLoopController, PlanLoopPolicy
from intergrax.runtime.drop_in_knowledge_mode.planning.step_planner import StepPlanner
from intergrax.runtime.drop_in_knowledge_mode.planning.step_executor import StepExecutor
from intergrax.runtime.drop_in_knowledge_mode.responses.response_schema import RuntimeAnswer
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.contract import RuntimeStepRunner
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.setup_steps_tool import SETUP_STEPS


class PlannerDynamicPipeline(RuntimePipeline):
    """
    DYNAMIC plan pipeline:
      - deterministic setup outside planner
      - EnginePlanner -> StepPlanner(DYNAMIC) -> StepExecutor(registry)
      - PlanLoopController.run_dynamic does iterative: plan -> execute -> iterate
      - runtime_answer MUST be produced by PersistAndBuildAnswerStep (or HITL wrapper in loop)
    """

    async def _inner_run(self, state: RuntimeState) -> RuntimeAnswer:
        # ---------------------------------------------------------------------
        # 0) Validate configuration early (fail fast, before planning/execution)
        # ---------------------------------------------------------------------
        ctx = state.context
        cfg = ctx.config
        req = state.request

        if cfg.llm_adapter is None:
            raise RuntimeError("PlannerDynamicPipeline: config.llm_adapter is required for EnginePlanner.")
        if cfg.step_planner_cfg is None:
            raise RuntimeError("PlannerDynamicPipeline: config.step_planner_cfg is required.")
        if cfg.step_executor_cfg is None:
            raise RuntimeError("PlannerDynamicPipeline: config.step_executor_cfg is required.")
        if cfg.plan_source is None and cfg.planner_prompt_config is None:
            raise RuntimeError(
                "PlannerDynamicPipeline: config.planner_prompt_config is required when config.plan_source is not set."
            )

        # ---------------------------------------------------------------------
        # 1) Deterministic setup outside planner (same category as NoPlannerPipeline setup)
        # ---------------------------------------------------------------------
        await RuntimeStepRunner.execute_pipeline(SETUP_STEPS, state)

        # ---------------------------------------------------------------------
        # 2) Build components from config/state (no new protocols/classes)
        # ---------------------------------------------------------------------
        engine_planner = EnginePlanner(
            llm_adapter=cfg.llm_adapter,
            plan_source=cfg.plan_source
        )
        
        step_planner = StepPlanner(cfg.step_planner_cfg)

        registry = self.build_default_planning_step_registry()
        step_executor = StepExecutor(registry=registry, cfg=cfg.step_executor_cfg)

        # ---------------------------------------------------------------------
        # 3) Plan -> Execute via PlanLoopController (DYNAMIC)
        # ---------------------------------------------------------------------
        loop = PlanLoopController(
            engine_planner=engine_planner,
            step_planner=step_planner,
            step_executor=step_executor,
            policy=cfg.plan_loop_policy or PlanLoopPolicy(),
        )

        ans = await loop.run_dynamic(
            state=state,
            plan_id_prefix=f"dynamic-{state.run_id}",
            user_message=req.message or "",
        )

        return ans
