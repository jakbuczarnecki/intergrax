# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import hashlib
from typing import Optional

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.engine_plan_models import EnginePlan, PlannerPromptConfig
from intergrax.runtime.nexus.planning.engine_planner_messages import EnginePlannerMessageBuilder
from intergrax.runtime.nexus.planning.engine_planner_parse import EnginePlanJsonParser
from intergrax.runtime.nexus.planning.plan_sources import PlanSourceMeta
from intergrax.runtime.nexus.planning.step_executor_models import ReplanContext
from intergrax.runtime.nexus.tracing.plan.capability_clamp import PlannerCapabilityClampDiagV1
from intergrax.runtime.nexus.tracing.plan.engine_plan_produced import PlannerEnginePlanProducedDiagV1
from intergrax.runtime.nexus.tracing.plan.plan_source_contract_violation import (
    PlannerPlanSourceContractViolationDiagV1,
)
from intergrax.runtime.nexus.tracing.plan.plan_source_failed import PlannerPlanSourceFailedDiagV1
from intergrax.runtime.nexus.tracing.plan.planner_build_debug import PlannerBuildDebugDiagV1
from intergrax.runtime.nexus.tracing.plan.raw_plan_parse_failed import PlannerRawPlanParseFailedDiagV1
from intergrax.runtime.events.planner_events import record_plan_failed
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class EnginePlannerDiagnostics:
    """Typed traces, parse safety, and capability clamp for EnginePlanner."""

    RAW_PREVIEW_LIMIT = 200
    RAW_TAIL_PREVIEW_LIMIT = 200

    def safe_parse_plan(
        self,
        *,
        raw: str,
        meta: Optional[PlanSourceMeta],
        prompt_config: Optional[PlannerPromptConfig],
        state: RuntimeState,
        config: RuntimeConfig,
    ) -> EnginePlan:
        try:
            return EnginePlanJsonParser.parse(raw, prompt_config=prompt_config)
        except (ValueError, TypeError) as e:
            self.trace_parse_failed(
                state=state,
                config=config,
                meta=meta,
                raw=raw,
                error=e,
            )
            raise

    def build_planner_debug(
        self,
        *,
        raw_text: str,
        meta: Optional[PlanSourceMeta],
        forced_plan_json_len: Optional[int],
        forced_plan_hash: Optional[str],
        replan_ctx: Optional[ReplanContext],
        raw_hash16: Optional[str] = None,
    ) -> PlannerBuildDebugDiagV1:
        raw_hash = raw_hash16 or hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
        replan_json = EnginePlannerMessageBuilder.serialize_replan_ctx(replan_ctx)
        replan_hash = (
            hashlib.sha256(replan_json.encode("utf-8")).hexdigest()[:16]
            if replan_json
            else None
        )

        return PlannerBuildDebugDiagV1(
            planner_forced_plan_used=(forced_plan_hash is not None),
            planner_source_kind=(meta.source_kind if meta else None),
            planner_source_detail=(meta.source_detail if meta else None),
            planner_replan_ctx_present=(replan_ctx is not None),
            planner_replan_ctx_hash=replan_hash,
            planner_raw_len=len(raw_text),
            planner_raw_hash=raw_hash,
            planner_raw_preview=raw_text[: self.RAW_PREVIEW_LIMIT],
            planner_raw_tail_preview=raw_text[-self.RAW_TAIL_PREVIEW_LIMIT :],
            planner_forced_plan_json_len=forced_plan_json_len,
            planner_forced_plan_hash=forced_plan_hash,
        )

    def trace_plan_produced(
        self,
        *,
        state: RuntimeState,
        plan: EnginePlan,
        planner_build_debug: Optional[PlannerBuildDebugDiagV1] = None,
    ) -> None:
        state.trace_event(
            component=TraceComponent.PLANNER,
            step="plan",
            message="Planner produced engine plan.",
            level=TraceLevel.INFO,
            payload=PlannerEnginePlanProducedDiagV1(
                intent=plan.intent.value,
                next_step=plan.next_step.value if plan.next_step is not None else None,
            ),
        )
        if planner_build_debug is not None:
            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan",
                message="Planner build debug.",
                level=TraceLevel.DEBUG,
                payload=planner_build_debug,
            )

    def trace_plansource_failed(
        self,
        *,
        state: RuntimeState,
        config: RuntimeConfig,
        plan_source_type: str,
        error: Exception,
    ) -> None:
        record_plan_failed(
            None,
            config=config,
            state=state,
            error=error,
            failure_kind="plan_source",
        )
        state.trace_event(
            component=TraceComponent.PLANNER,
            step="engine_planner",
            message="PlanSource failed while generating raw plan.",
            level=TraceLevel.ERROR,
            payload=PlannerPlanSourceFailedDiagV1(
                plan_source_type=plan_source_type,
                error_type=type(error).__name__,
                error_message=str(error),
            ),
        )

    def trace_plansource_contract_violation(
        self,
        *,
        state: RuntimeState,
        plan_source_type: str,
        raw: object,
    ) -> None:
        state.trace_event(
            component=TraceComponent.PLANNER,
            step="engine_planner",
            message="PlanSource contract violation: raw plan is not a string.",
            level=TraceLevel.ERROR,
            payload=PlannerPlanSourceContractViolationDiagV1(
                plan_source_type=plan_source_type,
                raw_type=type(raw).__name__,
            ),
        )

    def validate_against_capabilities(self, *, plan: EnginePlan, state: RuntimeState) -> EnginePlan:
        before_use_web = bool(plan.use_websearch)
        before_use_ltm = bool(plan.use_user_longterm_memory)
        before_use_rag = bool(plan.use_rag)
        before_use_tools = bool(plan.use_tools)

        if plan.use_websearch and not state.cap_websearch_available:
            plan.use_websearch = False
        if plan.use_user_longterm_memory and not state.cap_user_ltm_available:
            plan.use_user_longterm_memory = False
        if plan.use_rag and not state.cap_rag_available:
            plan.use_rag = False
        if plan.use_tools and not state.cap_tools_available:
            plan.use_tools = False

        state.trace_event(
            component=TraceComponent.PLANNER,
            step="capability_clamp",
            message="Planner capability clamp applied.",
            level=TraceLevel.DEBUG,
            payload=PlannerCapabilityClampDiagV1(
                before_use_websearch=before_use_web,
                before_use_user_longterm_memory=before_use_ltm,
                before_use_rag=before_use_rag,
                before_use_tools=before_use_tools,
                available_websearch=bool(state.cap_websearch_available),
                available_user_ltm=bool(state.cap_user_ltm_available),
                available_rag=bool(state.cap_rag_available),
                available_tools=bool(state.cap_tools_available),
                after_use_websearch=bool(plan.use_websearch),
                after_use_user_longterm_memory=bool(plan.use_user_longterm_memory),
                after_use_rag=bool(plan.use_rag),
                after_use_tools=bool(plan.use_tools),
            ),
        )
        return plan

    def trace_parse_failed(
        self,
        *,
        state: RuntimeState,
        config: RuntimeConfig,
        meta: Optional[PlanSourceMeta],
        raw: str,
        error: Exception,
    ) -> None:
        raw_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        record_plan_failed(
            None,
            config=config,
            state=state,
            error=error,
            failure_kind="parse",
            raw_hash=raw_hash,
        )
        state.trace_event(
            component=TraceComponent.PLANNER,
            step="engine_planner",
            message="Failed to parse raw plan.",
            level=TraceLevel.ERROR,
            payload=PlannerRawPlanParseFailedDiagV1(
                planner_source_kind=(meta.source_kind if meta is not None else None),
                planner_source_detail=(meta.source_detail if meta is not None else None),
                raw_len=len(raw),
                raw_hash=raw_hash,
                raw_preview=raw[: self.RAW_PREVIEW_LIMIT],
                raw_tail_preview=raw[-self.RAW_TAIL_PREVIEW_LIMIT :],
                error_type=type(error).__name__,
                error_message=str(error),
            ),
        )
