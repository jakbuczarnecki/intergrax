# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional

from intergrax.runtime.nexus.planning.engine_plan_models import EngineNextStep, EnginePlan
from intergrax.runtime.nexus.planning.stepplan_models import (
    EngineHints,
    ExecutionPlan,
    ExecutionStep,
    PlanBuildMode,
    PlanIntent,
    PlanMode,
    StepId,
)

from intergrax.runtime.nexus.planning.step_planner.assembly import StepPlanAssembly
from intergrax.runtime.nexus.planning.step_planner.config import StepPlannerConfig
from intergrax.runtime.nexus.planning.step_planner.step_factory import StepPlanStepFactory
from intergrax.runtime.nexus.tools.catalog_dispatch import catalog_tool_ids


class StepPlanStrategies:
    """Intent-specific execution plan builders (deterministic, no LLM)."""

    def __init__(
        self,
        cfg: StepPlannerConfig,
        factory: StepPlanStepFactory,
        assembly: StepPlanAssembly,
    ) -> None:
        self._cfg = cfg
        self._factory = factory
        self._assembly = assembly

    def build_from_engine_plan(
        self,
        *,
        user_message: str,
        engine_plan: EnginePlan,
        plan_id: Optional[str] = None,
        build_mode: PlanBuildMode = PlanBuildMode.STATIC,
    ) -> ExecutionPlan:
        """
        Adapter entrypoint: EnginePlanner -> StepPlanner.

        STATIC:
        - build full sequence using EngineHints (web/ltm/rag/tools -> draft -> verify -> finalize)

        DYNAMIC:
        - build a single-step plan based on engine_plan.next_step (ready for planning loop)
        """
        if engine_plan is None:
            raise ValueError("engine_plan is required")

        msg = (user_message or "").strip()
        pid = (plan_id or self._assembly.new_plan_id()).strip() or self._assembly.new_plan_id()

        hints = self._hints_from_engine_plan(engine_plan)
        intent = hints.intent or PlanIntent.GENERIC

        # Preserve upstream clarifying question if provided.
        # CLARIFY is always an EXECUTE plan that stops via HITL (no FINALIZE required),
        # regardless of STATIC/DYNAMIC build_mode.
        if intent == PlanIntent.CLARIFY:
            q = (engine_plan.clarifying_question or "").strip()
            if not q:
                q = self._assembly.clarifying_question(msg)

            return self._plan_clarify_execute_with_question(msg, plan_id=pid, question=q)

        if build_mode == PlanBuildMode.STATIC:
            return self.build_from_hints(
                user_message=msg,
                engine_hints=hints,
                plan_id=pid,
            )

        # DYNAMIC: one-step plan based on engine_plan.next_step
        ns = engine_plan.next_step
        if ns is None:
            # Fallback: if model didn't provide next_step, behave like STATIC
            return self.build_from_hints(
                user_message=msg,
                engine_hints=hints,
                plan_id=pid,
            )

        steps: List[ExecutionStep]

        if ns == EngineNextStep.WEBSEARCH:
            steps = [
                self._factory.websearch(
                    step_id=StepId.WEBSEARCH,
                    depends_on=[],
                    query=self._assembly.web_query(msg, intent=intent),
                )
            ]

        elif ns == EngineNextStep.TOOLS:
            steps = [
                self._factory.tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(intent)},
                    allowed_tool_ids=list(catalog_tool_ids(engine_plan.resolved_tool_ids())),
                    max_tool_calls=1,
                )
            ]

        elif ns == EngineNextStep.RAG:
            steps = [
                self._factory.rag_retrieval(
                    query=msg,
                    step_id=StepId.RAG,
                    depends_on=[],
                    top_k=6,
                )
            ]

        elif ns == EngineNextStep.SYNTHESIZE:
            steps = [
                self._factory.synthesize(
                    step_id=StepId.DRAFT,
                    depends_on=[],
                    instructions=msg,
                )
            ]

        elif ns == EngineNextStep.FINALIZE:
            steps = [
                self._factory.finalize(
                    depends_on=[],
                    instructions=msg,
                )
            ]

        else:
            # ns == CLARIFY is handled above via intent == CLARIFY
            steps = [
                self._factory.synthesize(
                    step_id=StepId.DRAFT,
                    depends_on=[],
                    instructions=msg,
                )
            ]

        # DYNAMIC plans do NOT have to end with FINALIZE_ANSWER.
        return self._assembly.wrap(
            intent=intent,
            mode=PlanMode.ITERATE,
            steps=steps,
            plan_id=pid,
            enforce_finalize=False,
        )




    def _hints_from_engine_plan(self, plan: EnginePlan) -> EngineHints:
        return EngineHints(
            enable_websearch=bool(plan.use_websearch),
            enable_ltm=bool(plan.use_user_longterm_memory),
            enable_rag=bool(plan.use_rag),
            enable_tools=bool(plan.use_tools),
            planner_tool_ids=catalog_tool_ids(plan.resolved_tool_ids()),
            intent=plan.intent,
            intent_reason=(plan.reasoning_summary or None),
        )
    

    def _plan_clarify_execute_with_question(self, msg: str, *, plan_id: str, question: str) -> ExecutionPlan:
        steps: List[ExecutionStep] = [
            self._factory.clarify(step_id=StepId.CLARIFY, depends_on=[], question=question),
        ]
        return self._assembly.wrap(
            plan_id=plan_id,
            intent=PlanIntent.CLARIFY,
            mode=PlanMode.EXECUTE,
            steps=steps,
            enforce_finalize=False,
        )



    def build_from_hints(
        self,
        *,
        user_message: str,
        engine_hints: Optional[EngineHints] = None,
        plan_id: Optional[str] = None,
    ) -> ExecutionPlan:
        msg = (user_message or "").strip()
        hints = engine_hints or EngineHints()
        pid = (plan_id or "stepplan-001").strip() or "stepplan-001"

        # If no message -> clarify (hard deterministic)
        if not msg:
            return self._plan_clarify(msg, plan_id=pid)

        # PRIMARY: upstream route decides.
        intent = hints.intent or PlanIntent.GENERIC

        if intent == PlanIntent.CLARIFY:
            return self._plan_clarify(msg, plan_id=pid)

        if intent == PlanIntent.FRESHNESS:
            # If upstream asked for freshness but websearch disabled -> degrade safely
            if hints.enable_websearch:
                return self._plan_freshness_with_hints(msg, plan_id=pid, hints=hints)
            return self._plan_generic_with_hints(msg, plan_id=pid, hints=hints)

        if intent == PlanIntent.PROJECT_ARCHITECTURE:
            if hints.enable_ltm:
                return self._plan_project_with_hints(msg, plan_id=pid, hints=hints)
            return self._plan_generic_with_hints(msg, plan_id=pid, hints=hints)


        # GENERIC default
        return self._plan_generic_with_hints(msg, plan_id=pid, hints=hints)


    def _plan_freshness_with_hints(self, msg: str, *, plan_id: str, hints: EngineHints) -> ExecutionPlan:
        pre_steps: List[ExecutionStep] = []

        # 1) WEBSEARCH must be first for freshness (if enabled)
        if hints.enable_websearch:
            pre_steps.append(
                self._factory.websearch(
                    step_id=StepId.WEBSEARCH,
                    depends_on=[],
                    query=self._assembly.web_query(msg, intent=PlanIntent.FRESHNESS),
                )
            )

        # 2) Optional RAG (after websearch, before draft)
        if hints.enable_rag:
            pre_steps.append(
                self._factory.rag_retrieval(
                    query=msg,
                    step_id=StepId.RAG,
                    depends_on=[],
                    top_k=6,
                )
            )

        # 3) Optional TOOLS
        if hints.enable_tools:
            pre_steps.append(
                self._factory.tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(PlanIntent.FRESHNESS)},
                    allowed_tool_ids=list(hints.planner_tool_ids),
                    max_tool_calls=1,
                )
            )

        pre_steps = self._assembly.chain_pre_steps(pre_steps)
        draft_deps = [pre_steps[-1].step_id] if pre_steps else []
        steps = pre_steps + self._assembly.build_execute_tail(msg=msg, depends_on=draft_deps)

        return self._assembly.wrap(plan_id=plan_id, intent=PlanIntent.FRESHNESS, mode=PlanMode.EXECUTE, steps=steps)


    def _plan_generic_with_hints(self, msg: str, *, plan_id: str, hints: EngineHints) -> ExecutionPlan:
        pre_steps: List[ExecutionStep] = []

        # Deterministic, conservative ordering for pre-draft:
        # RAG -> TOOLS (web/ltm are handled by dedicated plans)
        if hints.enable_rag:
            pre_steps.append(
                self._factory.rag_retrieval(query=msg, step_id=StepId.RAG, depends_on=[], top_k=6)
            )

        if hints.enable_tools:
            pre_steps.append(
                self._factory.tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(PlanIntent.GENERIC)},
                    allowed_tool_ids=list(hints.planner_tool_ids),
                    max_tool_calls=1,
                )
            )

        pre_steps = self._assembly.chain_pre_steps(pre_steps)
        draft_deps = [pre_steps[-1].step_id] if pre_steps else []
        steps = pre_steps + self._assembly.build_execute_tail(msg=msg, depends_on=draft_deps)

        return self._assembly.wrap(plan_id=plan_id, intent=PlanIntent.GENERIC, mode=PlanMode.EXECUTE, steps=steps)


    def _plan_project_with_hints(self, msg: str, *, plan_id: str, hints: EngineHints) -> ExecutionPlan:
        pre_steps: List[ExecutionStep] = []

        # 1) LTM must be first for project architecture (if enabled)
        if hints.enable_ltm:
            pre_steps.append(
                self._factory.ltm(
                    step_id=StepId.LTM_SEARCH,
                    depends_on=[],
                    query=self._assembly.ltm_query(msg, intent=PlanIntent.PROJECT_ARCHITECTURE),
                )
            )

        # 2) Optional RAG
        if hints.enable_rag:
            pre_steps.append(
                self._factory.rag_retrieval(
                    query=msg,
                    step_id=StepId.RAG,
                    depends_on=[],
                    top_k=6,
                )
            )

        # 3) Optional TOOLS
        if hints.enable_tools:
            pre_steps.append(
                self._factory.tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(PlanIntent.PROJECT_ARCHITECTURE)},
                    allowed_tool_ids=list(hints.planner_tool_ids),
                    max_tool_calls=1,
                )
            )

        pre_steps = self._assembly.chain_pre_steps(pre_steps)
        draft_deps = [pre_steps[-1].step_id] if pre_steps else []
        steps = pre_steps + self._assembly.build_execute_tail(msg=msg, depends_on=draft_deps)

        return self._assembly.wrap(plan_id=plan_id, intent=PlanIntent.PROJECT_ARCHITECTURE, mode=PlanMode.EXECUTE, steps=steps)


    def _plan_generic(self, msg: str, *, plan_id: str) -> ExecutionPlan:
        return self._plan_generic_with_hints(msg, plan_id=plan_id, hints=EngineHints())


    def _plan_freshness(self, msg: str, *, plan_id: str) -> ExecutionPlan:
        return self._plan_freshness_with_hints(msg, plan_id=plan_id, hints=EngineHints(enable_websearch=True, intent=PlanIntent.FRESHNESS))


    def _plan_project(self, msg: str, *, plan_id: str) -> ExecutionPlan:
        # Backward-compatible wrapper: "classic" project plan = LTM only.
        return self._plan_project_with_hints(
            msg,
            plan_id=plan_id,
            hints=EngineHints(enable_ltm=True, intent=PlanIntent.PROJECT_ARCHITECTURE),
        )


    def _plan_clarify(self, msg: str, *, plan_id: str) -> ExecutionPlan:
        q = self._assembly.clarifying_question(msg)
        return self._plan_clarify_execute_with_question(msg, plan_id=plan_id, question=q)


    

