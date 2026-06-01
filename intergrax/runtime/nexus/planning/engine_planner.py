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
from intergrax.runtime.nexus.planning.engine_planner_messages import EnginePlannerMessageBuilder
from intergrax.runtime.nexus.planning.engine_planner_parse import EnginePlanJsonParser
from intergrax.runtime.nexus.planning.plan_sources import LLMPlanSource, PlanRequest, PlanSource, PlanSourceMeta
from intergrax.runtime.nexus.planning.step_executor_models import ReplanContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.tracing.plan.capability_clamp import PlannerCapabilityClampDiagV1
from intergrax.runtime.nexus.tracing.plan.engine_plan_produced import PlannerEnginePlanProducedDiagV1
from intergrax.runtime.nexus.tracing.plan.plan_source_contract_violation import PlannerPlanSourceContractViolationDiagV1
from intergrax.runtime.nexus.tracing.plan.plan_source_failed import PlannerPlanSourceFailedDiagV1
from intergrax.runtime.nexus.tracing.plan.planner_build_debug import PlannerBuildDebugDiagV1
from intergrax.runtime.nexus.tracing.plan.raw_plan_parse_failed import PlannerRawPlanParseFailedDiagV1
from intergrax.runtime.events.planner_events import record_plan_failed
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel



class EnginePlanner:
    """
    LLM-based planner that outputs a typed EnginePlan.

    IMPORTANT:
    - No heuristics: the LLM decides based on PlannerInput + capabilities.
    - Output must be JSON only (we parse & validate).
    """

    _RAW_PREVIEW_LIMIT = 200
    _RAW_TAIL_PREVIEW_LIMIT = 200

    def __init__(
        self,
        *,
        llm_adapter: LLMAdapter,
        plan_source: Optional[PlanSource] = None,
    ) -> None:
        self._llm_adapter = llm_adapter
        self._plan_source: PlanSource = plan_source or LLMPlanSource()

        # Fail-fast contract (ABC)
        if not isinstance(self._plan_source, PlanSource):
            raise TypeError(
                f"plan_source must be a PlanSource, got: {type(self._plan_source).__name__}"
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

        # ---------------------------------------------------------------------
        # Deterministic override (production feature):
        # - replay a captured plan
        # - run planning without LLM (offline / incident mode)
        # ---------------------------------------------------------------------
        forced_plan = None
        planner_build_debug: Optional[PlannerBuildDebugDiagV1] = None

        if prompt_config is not None:
            forced_plan = prompt_config.forced_plan

        if forced_plan is not None:
            if isinstance(forced_plan, EnginePlan):
                forced_plan_dict = forced_plan.to_planner_dict()
            elif isinstance(forced_plan, dict):
                forced_plan_dict = forced_plan
            else:
                raise TypeError("forced_plan must be EnginePlan or dict")

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

            plan = self._safe_parse_plan(
                raw=forced_json,
                meta=meta_forced,
                prompt_config=prompt_config,
                state=state,
                config=config,
            )

            # Capability clamp (keep semantics consistent with LLM-based planning)
            plan = self._validate_against_capabilities(plan=plan, state=state)

            forced_hash = hashlib.sha256(forced_json.encode("utf-8")).hexdigest()[:16]

            planner_build_debug = self._build_planner_debug(
                raw_text=forced_json,
                meta=meta_forced,  # FIX: was meta (undefined/wrong)
                forced_plan_json_len=len(forced_json),
                forced_plan_hash=forced_hash,
                replan_ctx=replan_ctx,
                raw_hash16=None,
            )

            self._trace_plan_produced(
                state=state,
                plan=plan,
                planner_build_debug=planner_build_debug,
            )
            return plan

        # ---------------------------------------------------------------------
        # Normal LLM planning flow
        # ---------------------------------------------------------------------
        messages = self._build_planner_messages(
            req=req,
            state=state,
            config=config,
            prompt_config=prompt_config,
            replan_ctx=replan_ctx,
        )

        try:
            req_ps = PlanRequest(
                llm_adapter=self._llm_adapter,
                messages=messages,
                run_id=run_id,
                replan_ctx=replan_ctx,
            )

            res = await self._plan_source.generate_plan_raw(req=req_ps)
            raw = res.raw
            meta = res.meta
        except Exception as e:
            self._trace_plansource_failed(state=state, config=config, error=e)
            raise RuntimeError(
                f"PlanSource failed: {type(self._plan_source).__name__}: {type(e).__name__}: {e}"
            ) from e

        if not isinstance(raw, str):
            # This should never happen; PlanSource contract violation
            self._trace_plansource_contract_violation(state=state, raw=raw)
            raise TypeError(
                f"PlanSource contract violation: expected str raw plan, got {type(raw).__name__}"
            )

        if meta is None:
            # allow meta to be optional, but normalize it
            meta = PlanSourceMeta(source_kind="unknown", source_detail=type(self._plan_source).__name__)

        plan = self._safe_parse_plan(
            raw=raw,
            meta=meta,
            prompt_config=prompt_config,
            state=state,
            config=config,
        )

        plan = self._validate_against_capabilities(plan=plan, state=state)

        planner_build_debug = self._build_planner_debug(
            raw_text=raw,
            meta=meta,
            forced_plan_json_len=None,
            forced_plan_hash=None,
            replan_ctx=replan_ctx,
            raw_hash16=res.raw_hash16,
        )

        self._trace_plan_produced(
            state=state,
            plan=plan,
            planner_build_debug=planner_build_debug,
        )

        return plan

    

    def _safe_parse_plan(
        self,
        *,
        raw: str,
        meta: Optional[PlanSourceMeta],
        prompt_config: Optional[PlannerPromptConfig],
        state: RuntimeState,
        config: RuntimeConfig,
    ) -> EnginePlan:
        try:
            return self._parse_plan(raw, prompt_config=prompt_config)
        except (ValueError, TypeError) as e:
            self._trace_parse_failed(
                state=state,
                config=config,
                meta=meta,
                raw=raw,
                error=e,
            )
            raise

    def _build_planner_debug(
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
        replan_json = self._serialize_replan_ctx(replan_ctx)
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
            planner_raw_preview=raw_text[: self._RAW_PREVIEW_LIMIT],
            planner_raw_tail_preview=raw_text[-self._RAW_TAIL_PREVIEW_LIMIT :],

            planner_forced_plan_json_len=forced_plan_json_len,
            planner_forced_plan_hash=forced_plan_hash,
        )


    def _trace_plan_produced(
        self,
        *,
        state: RuntimeState,
        plan: EnginePlan,
        planner_build_debug: Optional[PlannerBuildDebugDiagV1] = None,
    ) -> None:
        # Existing "plan produced" trace (bez dict debug)
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

        # Separate typed debug trace (recommended)
        if planner_build_debug is not None:
            state.trace_event(
                component=TraceComponent.PLANNER,
                step="plan",
                message="Planner build debug.",
                level=TraceLevel.DEBUG,
                payload=planner_build_debug,
            )


    
    def _trace_plansource_failed(
        self,
        *,
        state: RuntimeState,
        config: RuntimeConfig,
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
                plan_source_type=type(self._plan_source).__name__,
                error_type=type(error).__name__,
                error_message=str(error),
            ),
        )

    def _trace_plansource_contract_violation(
        self,
        *,
        state: RuntimeState,
        raw: object,
    ) -> None:
        state.trace_event(
            component=TraceComponent.PLANNER,
            step="engine_planner",
            message="PlanSource contract violation: raw plan is not a string.",
            level=TraceLevel.ERROR,
            payload=PlannerPlanSourceContractViolationDiagV1(
                plan_source_type=type(self._plan_source).__name__,
                raw_type=type(raw).__name__,
            ),
        )
    
    def _validate_against_capabilities(self, *, plan: EnginePlan, state: RuntimeState) -> EnginePlan:
        """
        Hard capability clamp. No heuristics, no intent changes.
        Only disables flags that are not available in the current runtime.
        """

        before_use_web = bool(plan.use_websearch)
        before_use_ltm = bool(plan.use_user_longterm_memory)
        before_use_rag = bool(plan.use_rag)
        before_use_tools = bool(plan.use_tools)

        available_web = bool(state.cap_websearch_available)
        available_ltm = bool(state.cap_user_ltm_available)
        available_rag = bool(state.cap_rag_available)
        available_tools = bool(state.cap_tools_available)

        if plan.use_websearch and not state.cap_websearch_available:
            plan.use_websearch = False

        if plan.use_user_longterm_memory and not state.cap_user_ltm_available:
            plan.use_user_longterm_memory = False

        if plan.use_rag and not state.cap_rag_available:
            plan.use_rag = False

        if plan.use_tools and not state.cap_tools_available:
            plan.use_tools = False

        after_use_web = bool(plan.use_websearch)
        after_use_ltm = bool(plan.use_user_longterm_memory)
        after_use_rag = bool(plan.use_rag)
        after_use_tools = bool(plan.use_tools)

        # Emit typed clamp diagnostics (no dicts, no plan.debug)
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

                available_websearch=available_web,
                available_user_ltm=available_ltm,
                available_rag=available_rag,
                available_tools=available_tools,

                after_use_websearch=after_use_web,
                after_use_user_longterm_memory=after_use_ltm,
                after_use_rag=after_use_rag,
                after_use_tools=after_use_tools,
            ),
        )

        return plan


    def _trace_parse_failed(
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
                raw_preview=raw[: self._RAW_PREVIEW_LIMIT],
                raw_tail_preview=raw[-self._RAW_TAIL_PREVIEW_LIMIT :],
                error_type=type(error).__name__,
                error_message=str(error),
            ),
        )


    # -----------------------------
    # Prompting
    # -----------------------------

    def _serialize_replan_ctx(self, replan_ctx: Optional[ReplanContext]) -> Optional[str]:
        return EnginePlannerMessageBuilder.serialize_replan_ctx(replan_ctx)

    def _build_planner_messages(
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



    def _extract_json_object(self, raw: str) -> str:
        return EnginePlanJsonParser.extract_json_object(raw)

    def _parse_plan(self, raw: str, *, prompt_config: Optional[PlannerPromptConfig]) -> EnginePlan:
        return EnginePlanJsonParser.parse(raw, prompt_config=prompt_config)

    def _fallback_clarify_question(self, prompt_config: Optional[PlannerPromptConfig]) -> str:
        return EnginePlanJsonParser.fallback_clarify_question(prompt_config)


    def _json_shape(self, obj: object) -> dict:
        """
        Return a lightweight structural summary of a JSON-like object.
        No values, only types and key presence. Production-safe for traces.
        """
        if isinstance(obj, dict):
            # Keep only a small subset of keys to avoid large traces
            keys = sorted(list(obj.keys()))
            return {
                "type": "object",
                "keys_count": len(keys),
                "keys_preview": keys[:30],
            }
        if isinstance(obj, list):
            return {
                "type": "array",
                "len": len(obj),
            }
        return {"type": type(obj).__name__}
