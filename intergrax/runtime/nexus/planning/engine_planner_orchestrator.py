# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import hashlib
import json
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan, PlannerPromptConfig
from intergrax.runtime.nexus.planning.engine_planner_diagnostics import EnginePlannerDiagnostics
from intergrax.runtime.nexus.planning.engine_planner_messages import EnginePlannerMessageBuilder
from intergrax.runtime.nexus.planning.plan_sources import (
    LLMPlanSource,
    PlanRequest,
    PlanSource,
    PlanSourceMeta,
)
from intergrax.runtime.nexus.planning.step_executor_models import ReplanContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest


class EnginePlannerOrchestrator:
    """LLM plan generation and forced-plan replay (no tracing side-effects in public API)."""

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        plan_source: PlanSource,
        diagnostics: EnginePlannerDiagnostics,
    ) -> None:
        self._llm_adapter = llm_adapter
        self._plan_source = plan_source
        self._diag = diagnostics

    @classmethod
    def create(
        cls,
        *,
        llm_adapter: LLMAdapter,
        plan_source: Optional[PlanSource] = None,
    ) -> EnginePlannerOrchestrator:
        source = plan_source or LLMPlanSource()
        if not isinstance(source, PlanSource):
            raise TypeError(f"plan_source must be a PlanSource, got: {type(source).__name__}")
        return cls(
            llm_adapter=llm_adapter,
            plan_source=source,
            diagnostics=EnginePlannerDiagnostics(),
        )

    @property
    def plan_source(self) -> PlanSource:
        return self._plan_source

    @property
    def diagnostics(self) -> EnginePlannerDiagnostics:
        return self._diag

    async def plan_from_forced(
        self,
        *,
        forced_plan: EnginePlan | dict[str, object],
        state: RuntimeState,
        config: RuntimeConfig,
        prompt_config: Optional[PlannerPromptConfig],
        replan_ctx: Optional[ReplanContext],
    ) -> tuple[EnginePlan, object]:
        """Returns (plan, planner_build_debug)."""
        if isinstance(forced_plan, EnginePlan):
            forced_plan_dict = forced_plan.to_planner_dict()
        else:
            forced_plan_dict = forced_plan

        meta_forced = PlanSourceMeta(
            source_kind="forced",
            source_detail="prompt_config.forced_plan",
        )
        forced_json = json.dumps(
            forced_plan_dict,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        plan = self._diag.safe_parse_plan(
            raw=forced_json,
            meta=meta_forced,
            prompt_config=prompt_config,
            state=state,
            config=config,
        )
        plan = self._diag.validate_against_capabilities(plan=plan, state=state)
        forced_hash = hashlib.sha256(forced_json.encode("utf-8")).hexdigest()[:16]
        debug = self._diag.build_planner_debug(
            raw_text=forced_json,
            meta=meta_forced,
            forced_plan_json_len=len(forced_json),
            forced_plan_hash=forced_hash,
            replan_ctx=replan_ctx,
            raw_hash16=None,
        )
        return plan, debug

    async def plan_from_llm(
        self,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        config: RuntimeConfig,
        prompt_config: Optional[PlannerPromptConfig],
        replan_ctx: Optional[ReplanContext],
        run_id: Optional[str],
    ) -> tuple[EnginePlan, object]:
        """Returns (plan, planner_build_debug). Raises on PlanSource failure."""
        messages = self.build_planner_messages(
            req=req,
            state=state,
            config=config,
            prompt_config=prompt_config,
            replan_ctx=replan_ctx,
        )
        req_ps = PlanRequest(
            llm_adapter=self._llm_adapter,
            messages=messages,
            run_id=run_id,
            replan_ctx=replan_ctx,
        )
        res = await self._plan_source.generate_plan_raw(req=req_ps)
        raw = res.raw
        meta = res.meta

        if not isinstance(raw, str):
            self._diag.trace_plansource_contract_violation(
                state=state,
                plan_source_type=type(self._plan_source).__name__,
                raw=raw,
            )
            raise TypeError(
                f"PlanSource contract violation: expected str raw plan, got {type(raw).__name__}"
            )

        if meta is None:
            meta = PlanSourceMeta(
                source_kind="unknown",
                source_detail=type(self._plan_source).__name__,
            )

        plan = self._diag.safe_parse_plan(
            raw=raw,
            meta=meta,
            prompt_config=prompt_config,
            state=state,
            config=config,
        )
        plan = self._diag.validate_against_capabilities(plan=plan, state=state)
        debug = self._diag.build_planner_debug(
            raw_text=raw,
            meta=meta,
            forced_plan_json_len=None,
            forced_plan_hash=None,
            replan_ctx=replan_ctx,
            raw_hash16=res.raw_hash16,
        )
        return plan, debug

    def build_planner_messages(
        self,
        *,
        req: RuntimeRequest,
        state: RuntimeState,
        config: RuntimeConfig,
        prompt_config: Optional[PlannerPromptConfig] = None,
        replan_ctx: Optional[ReplanContext] = None,
    ) -> list[ChatMessage]:
        _ = config
        return EnginePlannerMessageBuilder.build_messages(
            req=req,
            state=state,
            prompt_config=prompt_config,
            replan_ctx=replan_ctx,
        )
