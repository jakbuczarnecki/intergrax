# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Optional

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan, PlannerPromptConfig
from intergrax.runtime.nexus.planning.engine_planner_orchestrator import EnginePlannerOrchestrator
from intergrax.runtime.nexus.planning.plan_sources import PlanSource
from intergrax.runtime.nexus.planning.step_executor_models import ReplanContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


class EnginePlanner:
    """
    LLM-based planner that outputs a typed EnginePlan.

    IMPORTANT:
    - No heuristics: the LLM decides based on PlannerInput + capabilities.
    - Output must be JSON only (we parse & validate).
    """

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        plan_source: Optional[PlanSource] = None,
    ) -> None:
        self._orchestrator = EnginePlannerOrchestrator.create(
            llm_adapter=llm_adapter,
            plan_source=plan_source,
        )

    async def plan(
        self,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        config: RuntimeConfig,
        prompt_config: Optional[PlannerPromptConfig] = None,
        run_id: Optional[str] = None,
        replan_ctx: Optional[ReplanContext] = None,
    ) -> EnginePlan:
        forced_plan = prompt_config.forced_plan if prompt_config is not None else None

        if forced_plan is not None:
            if not isinstance(forced_plan, (EnginePlan, dict)):
                raise TypeError("forced_plan must be EnginePlan or dict")
            plan, debug = await self._orchestrator.plan_from_forced(
                forced_plan=forced_plan,
                state=state,
                config=config,
                prompt_config=prompt_config,
                replan_ctx=replan_ctx,
            )
            self._orchestrator.diagnostics.trace_plan_produced(
                state=state,
                plan=plan,
                planner_build_debug=debug,
            )
            return plan

        try:
            plan, debug = await self._orchestrator.plan_from_llm(
                req=req,
                state=state,
                config=config,
                prompt_config=prompt_config,
                replan_ctx=replan_ctx,
                run_id=run_id,
            )
        except (ValueError, TypeError):
            # Parse/validation errors — already traced via EnginePlannerDiagnostics.
            raise
        except Exception as e:
            self._orchestrator.diagnostics.trace_plansource_failed(
                state=state,
                config=config,
                plan_source_type=type(self._orchestrator.plan_source).__name__,
                error=e,
            )
            raise RuntimeError(
                f"PlanSource failed: {type(self._orchestrator.plan_source).__name__}: "
                f"{type(e).__name__}: {e}"
            ) from e

        self._orchestrator.diagnostics.trace_plan_produced(
            state=state,
            plan=plan,
            planner_build_debug=debug,
        )
        return plan
