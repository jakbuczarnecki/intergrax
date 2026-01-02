# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import uuid

from intergrax.runtime.drop_in_knowledge_mode.planning.engine_plan_models import EngineNextStep, EnginePlan
from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import (
    EngineHints,
    ExecutionPlan,
    ExecutionStep,
    ExpectedOutputType,
    FailurePolicy,
    FailurePolicyKind,
    OutputFormat,
    PlanBudgets,
    PlanBuildMode,
    PlanIntent,
    PlanMode,
    RationaleType,
    StepAction,
    StepBudgets,
    StepId,
    StopConditions,
    VerifyCriterion,
    VerifySeverity,
    WebSearchStrategy,
)


@dataclass(frozen=True)
class StepPlannerConfig:
    """
    Rule-based step planner configuration.
    Keep it deterministic; no LLM prompting here.
    """

    # Output style
    final_answer_style: str = "concise_technical"
    final_format: OutputFormat = OutputFormat.MARKDOWN

    # Default per-step budgets
    step_max_chars: int = 2000

    web_top_k: int = 5
    web_max_results: int = 5
    web_recency_days: int = 30
    web_strategy: WebSearchStrategy = WebSearchStrategy.HYBRID

    # Plan-level budgets
    max_total_steps: int = 6
    max_total_tool_calls: int = 3
    max_total_web_queries: int = 5
    max_total_chars_context: int = 12000
    max_total_tokens_output: Optional[int] = None


class StepPlanner:
    """
    Deterministic planner that builds an ExecutionPlan from:
      - user_message
      - engine_hints (e.g. from engine_planner.py)

    Important: ExecutionStep.params MUST be a dict (per stepplan_models.ExecutionStep).
    """

    def __init__(self, cfg: Optional[StepPlannerConfig] = None):
        self._cfg = cfg or StepPlannerConfig()

    # -----------------------------
    # Public API
    # -----------------------------

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
        pid = (plan_id or self._new_plan_id()).strip() or self._new_plan_id()

        hints = self._hints_from_engine_plan(engine_plan)
        intent = hints.intent or PlanIntent.GENERIC

        # Preserve upstream clarifying question if provided
        if intent == PlanIntent.CLARIFY:
            q = (engine_plan.clarifying_question or "").strip()
            if not q:
                q = self._clarifying_question(msg)

            if build_mode == PlanBuildMode.STATIC:
                # STATIC must be an EXECUTE plan and end with FINAL
                return self._plan_clarify_execute_with_question(msg, plan_id=pid, question=q)

            # DYNAMIC must be ITERATE and can be single-step
            return self._plan_clarify_iterate_with_question(msg, plan_id=pid, question=q)

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
                self._step_websearch(
                    step_id=StepId.WEBSEARCH,
                    depends_on=[],
                    query=self._web_query(msg, intent=intent),
                )
            ]

        elif ns == EngineNextStep.TOOLS:
            steps = [
                self._step_tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(intent)},
                    max_tool_calls=1,
                )
            ]

        elif ns == EngineNextStep.RAG:
            steps = [
                self._step_rag_retrieval(
                    query=msg,
                    step_id=StepId.RAG,
                    depends_on=[],
                    top_k=6,
                )
            ]

        elif ns == EngineNextStep.SYNTHESIZE:
            # Use existing synth builder (do NOT call non-existent _step_synthesize_draft)
            steps = [
                self._step_synthesize(
                    step_id=StepId.DRAFT,
                    depends_on=[],
                    instructions=msg,
                )
            ]

        elif ns == EngineNextStep.FINALIZE:
            # Use existing finalize builder (do NOT call non-existent _step_finalize_answer)
            steps = [
                self._step_finalize(
                    depends_on=[],
                    instructions=msg,
                )
            ]

        else:
            # ns == CLARIFY was handled above via intent == CLARIFY
            # but keep a safe fallback:
            steps = [
                self._step_synthesize(
                    step_id=StepId.DRAFT,
                    depends_on=[],
                    instructions=msg,
                )
            ]

        # DYNAMIC plans do NOT have to end with FINALIZE_ANSWER.
        return self._wrap(
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
            intent=plan.intent,
            intent_reason=(plan.reasoning_summary or None),
        )
    

    def _chain_pre_steps(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
        """
        Ensure deterministic sequential execution for pre-steps:
        steps[0].depends_on stays as-is (expected empty),
        steps[i].depends_on = [steps[i-1].step_id] for i>0.
        """
        if not steps:
            return steps

        # First step: enforce no deps (pre-steps start the chain)
        steps[0].depends_on = []

        for i in range(1, len(steps)):
            steps[i].depends_on = [steps[i - 1].step_id]

        return steps
    
    def _plan_clarify_execute_with_question(self, msg: str, *, plan_id: str, question: str) -> ExecutionPlan:
        steps: List[ExecutionStep] = [
            self._step_clarify(step_id=StepId.CLARIFY, depends_on=[], question=question),
            # FINAL ensures a consistent "output step" and keeps invariant "plan ends with FINAL"
            self._step_finalize(depends_on=[StepId.CLARIFY], instructions=question),
        ]
        return self._wrap(
            plan_id=plan_id,
            intent=PlanIntent.CLARIFY,
            mode=PlanMode.EXECUTE,
            steps=steps,
            enforce_finalize=True,
        )

    def _plan_clarify_iterate_with_question(self, msg: str, *, plan_id: str, question: str) -> ExecutionPlan:
        steps: List[ExecutionStep] = [
            self._step_clarify(step_id=StepId.CLARIFY, depends_on=[], question=question),
        ]
        return self._wrap(
            plan_id=plan_id,
            intent=PlanIntent.CLARIFY,
            mode=PlanMode.ITERATE,
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


    def _build_execute_tail(
        self,
        *,
        msg: str,
        depends_on: List[StepId],
    ) -> List[ExecutionStep]:
        """
        Standard EXECUTE tail: DRAFT -> VERIFY -> FINAL.
        `depends_on` defines what DRAFT depends on (can be empty).
        """
        steps: List[ExecutionStep] = [
            self._step_synthesize(step_id=StepId.DRAFT, depends_on=depends_on, instructions=msg),
            self._step_verify(depends_on=[StepId.DRAFT], criteria=self._default_verify_criteria(msg), strict=True),
            self._step_finalize(depends_on=[StepId.VERIFY], instructions=msg),
        ]
        return steps


    # -----------------------------
    # Plan builders
    # -----------------------------

    def _plan_freshness_with_hints(self, msg: str, *, plan_id: str, hints: EngineHints) -> ExecutionPlan:
        pre_steps: List[ExecutionStep] = []

        # 1) WEBSEARCH must be first for freshness (if enabled)
        if hints.enable_websearch:
            pre_steps.append(
                self._step_websearch(
                    step_id=StepId.WEBSEARCH,
                    depends_on=[],
                    query=self._web_query(msg, intent=PlanIntent.FRESHNESS),
                )
            )

        # 2) Optional RAG (after websearch, before draft)
        if hints.enable_rag:
            pre_steps.append(
                self._step_rag_retrieval(
                    query=msg,
                    step_id=StepId.RAG,
                    depends_on=[],
                    top_k=6,
                )
            )

        # 3) Optional TOOLS
        if hints.enable_tools:
            pre_steps.append(
                self._step_tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(PlanIntent.FRESHNESS)},
                    max_tool_calls=1,
                )
            )

        pre_steps = self._chain_pre_steps(pre_steps)
        draft_deps = [pre_steps[-1].step_id] if pre_steps else []
        steps = pre_steps + self._build_execute_tail(msg=msg, depends_on=draft_deps)

        return self._wrap(plan_id=plan_id, intent=PlanIntent.FRESHNESS, mode=PlanMode.EXECUTE, steps=steps)


    def _plan_generic_with_hints(self, msg: str, *, plan_id: str, hints: EngineHints) -> ExecutionPlan:
        pre_steps: List[ExecutionStep] = []

        # Deterministic, conservative ordering for pre-draft:
        # RAG -> TOOLS (web/ltm are handled by dedicated plans)
        if hints.enable_rag:
            pre_steps.append(
                self._step_rag_retrieval(query=msg, step_id=StepId.RAG, depends_on=[], top_k=6)
            )

        if hints.enable_tools:
            pre_steps.append(
                self._step_tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(PlanIntent.GENERIC)},
                    max_tool_calls=1,
                )
            )

        pre_steps = self._chain_pre_steps(pre_steps)
        draft_deps = [pre_steps[-1].step_id] if pre_steps else []
        steps = pre_steps + self._build_execute_tail(msg=msg, depends_on=draft_deps)

        return self._wrap(plan_id=plan_id, intent=PlanIntent.GENERIC, mode=PlanMode.EXECUTE, steps=steps)


    def _plan_project_with_hints(self, msg: str, *, plan_id: str, hints: EngineHints) -> ExecutionPlan:
        pre_steps: List[ExecutionStep] = []

        # 1) LTM must be first for project architecture (if enabled)
        if hints.enable_ltm:
            pre_steps.append(
                self._step_ltm(
                    step_id=StepId.LTM_SEARCH,
                    depends_on=[],
                    query=self._ltm_query(msg, intent=PlanIntent.PROJECT_ARCHITECTURE),
                )
            )

        # 2) Optional RAG
        if hints.enable_rag:
            pre_steps.append(
                self._step_rag_retrieval(
                    query=msg,
                    step_id=StepId.RAG,
                    depends_on=[],
                    top_k=6,
                )
            )

        # 3) Optional TOOLS
        if hints.enable_tools:
            pre_steps.append(
                self._step_tools(
                    step_id=StepId.TOOLS,
                    depends_on=[],
                    tool_input={"query": msg, "intent": str(PlanIntent.PROJECT_ARCHITECTURE)},
                    max_tool_calls=1,
                )
            )

        pre_steps = self._chain_pre_steps(pre_steps)
        draft_deps = [pre_steps[-1].step_id] if pre_steps else []
        steps = pre_steps + self._build_execute_tail(msg=msg, depends_on=draft_deps)

        return self._wrap(plan_id=plan_id, intent=PlanIntent.PROJECT_ARCHITECTURE, mode=PlanMode.EXECUTE, steps=steps)


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
        q = self._clarifying_question(msg)
        return self._plan_clarify_execute_with_question(msg, plan_id=plan_id, question=q)


    

    def _plan_budgets(self) -> PlanBudgets:
        """
        Plan-level budgets. Deterministic defaults.
        Keep these small and stable; engine/runtime can override if needed.
        """
        return PlanBudgets(
            max_total_steps=self._cfg.max_total_steps,
            max_total_tool_calls=self._cfg.max_total_tool_calls,
            max_total_web_queries=self._cfg.max_total_web_queries,
            max_total_chars_context=self._cfg.max_total_chars_context,
            max_total_tokens_output=self._cfg.max_total_tokens_output,
        )


    def _stop_conditions(self, mode: PlanMode) -> StopConditions:
        if mode == PlanMode.ITERATE:
        # In iterative mode, we execute a single step and return to the planning loop.
            return StopConditions(
                max_iterations=1,
                stop_on_verifier_pass=False,
                stop_on_no_progress=True,
            )

        # EXECUTE (static full plan)
        return StopConditions(
            max_iterations=20,
            stop_on_verifier_pass=True,
            stop_on_no_progress=True,
        )


    def _web_query(self, msg: str, *, intent: PlanIntent) -> str:
        """
        Deterministic web query builder.
        IMPORTANT: routing decision (czy web w ogóle) nie jest tutaj.
        Tu tylko budujemy query, jeśli upstream już zdecydował, że websearch jest potrzebny.
        """
        q = (msg or "").strip()
        if not q:
            return "OpenAI Responses API changes"

        # Intent-specific normalization (no heuristics, just formatting)
        if intent == PlanIntent.FRESHNESS:
            # Keep it close to user text, but nudge toward changelog/release notes.
            return f"{q} changelog release notes dates"

        return q


    def _ltm_query(self, msg: str, *, intent: PlanIntent) -> str:
        """
        Deterministic LTM query builder.
        Again: no routing here; only build the query when LTM retrieval is already allowed.
        """
        q = (msg or "").strip()
        if not q:
            return "Intergrax architecture decisions"

        if intent == PlanIntent.PROJECT_ARCHITECTURE:
            # Keep stable prefix to improve retrieval consistency
            return f"Intergrax architecture: {q}"

        return q

    # -----------------------------
    # Step factories (IMPORTANT: params MUST be dict)
    # -----------------------------

    def _step_synthesize(self, *, step_id: StepId, depends_on: List[StepId], instructions: str) -> ExecutionStep:
        return ExecutionStep(
            step_id=step_id,
            action=StepAction.SYNTHESIZE_DRAFT,
            enabled=True,
            depends_on=depends_on,
            budgets=StepBudgets(top_k=0, max_chars=self._cfg.step_max_chars, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params={
                "instructions": instructions,
                "must_include": [],
                "avoid": [],
            },
            expected_output_type=ExpectedOutputType.DRAFT,
            rationale_type=RationaleType.PRODUCE_DRAFT,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.RETRY,
                max_retries=1,
                retry_backoff_ms=0,
                replan_reason=None,
            ),
        )

    def _step_verify(
        self,
        *,
        depends_on: List[StepId],
        criteria: List[VerifyCriterion],
        strict: bool,
    ) -> ExecutionStep:
        if not criteria:
            criteria = [VerifyCriterion(id="non_empty", description="Answer is non-empty", severity=VerifySeverity.ERROR)]

        return ExecutionStep(
            step_id=StepId.VERIFY,
            action=StepAction.VERIFY_ANSWER,
            enabled=True,
            depends_on=depends_on or [],
            budgets=StepBudgets(top_k=0, max_chars=1000, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params={
                # ExecutionStep expects dict; models will validate/normalize.
                "criteria": [c.model_dump() for c in criteria],
                "strict": bool(strict),
            },
            expected_output_type=ExpectedOutputType.VERIFIED,            
            rationale_type=RationaleType.VERIFY_QUALITY,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.REPLAN,
                max_retries=0,
                retry_backoff_ms=0,
                replan_reason="Verification failed",
            ),
        )

    def _step_finalize(self, *, depends_on: List[StepId], instructions: str) -> ExecutionStep:
        return ExecutionStep(
            step_id=StepId.FINAL,
            action=StepAction.FINALIZE_ANSWER,
            enabled=True,
            depends_on=depends_on,
            budgets=StepBudgets(top_k=0, max_chars=self._cfg.step_max_chars, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params={
                "instructions": instructions,
                "format": self._cfg.final_format.value,  # OutputFormat -> string for params model
            },
            expected_output_type=ExpectedOutputType.FINAL,
            rationale_type=RationaleType.FINALIZE,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.FAIL,
                max_retries=0,
                retry_backoff_ms=0,
                replan_reason=None,
            ),
        )

    def _step_websearch(self, *, step_id: StepId, depends_on: List[StepId], query: str) -> ExecutionStep:
        return ExecutionStep(
            step_id=step_id,
            action=StepAction.USE_WEBSEARCH,
            enabled=True,
            depends_on=depends_on,
            budgets=StepBudgets(top_k=self._cfg.web_top_k, max_chars=5000, max_tool_calls=0, max_web_queries=1),
            inputs={},
            params={
                "query": query,
                "recency_days": int(self._cfg.web_recency_days),
                "max_results": int(self._cfg.web_max_results),
                "strategy": self._cfg.web_strategy.value,  # WebSearchStrategy -> string for params model
                "domains_allowlist": None,
            },
            expected_output_type=ExpectedOutputType.SEARCH_RESULTS,
            rationale_type=RationaleType.RETRIEVE_WEB,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.REPLAN,
                max_retries=0,
                retry_backoff_ms=0,
                replan_reason="Web search failed",
            ),
        )

    def _step_ltm(self, *, step_id: StepId, depends_on: List[StepId], query: str) -> ExecutionStep:
        return ExecutionStep(
            step_id=step_id,
            action=StepAction.USE_USER_LONGTERM_MEMORY_SEARCH,
            enabled=True,
            depends_on=depends_on,
            budgets=StepBudgets(top_k=5, max_chars=2000, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params={
                "query": query,
                "top_k": 5,
                "score_threshold": None,
                "include_debug": False,
            },
            expected_output_type=ExpectedOutputType.LTM_RESULTS,
            rationale_type=RationaleType.RETRIEVE_LTM,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.RETRY,
                max_retries=1,
                retry_backoff_ms=0,
                replan_reason=None,
            ),
        )

    def _step_clarify(self, *, step_id: StepId, depends_on: List[StepId], question: str) -> ExecutionStep:
        # Clarify mode requires first step action ASK_CLARIFYING_QUESTION.
        return ExecutionStep(
            step_id=step_id,
            action=StepAction.ASK_CLARIFYING_QUESTION,
            enabled=True,
            depends_on=depends_on,
            budgets=StepBudgets(top_k=0, max_chars=300, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params={
                "question": question,
                "choices": None,
                "must_answer_to_continue": True,
            },
            expected_output_type=ExpectedOutputType.CLARIFYING_QUESTION,
            rationale_type=RationaleType.ASK_CLARIFICATION,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.FAIL,
                max_retries=0,
                retry_backoff_ms=0,
                replan_reason=None,
            ),
        )
    
    def _step_rag_retrieval(
        self,
        *,
        query: str,
        step_id: StepId = StepId.RAG,
        depends_on: Optional[List[StepId]] = None,
        top_k: int = 6,
    ) -> ExecutionStep:
        """
        Retrieve context from RAG vectorstore (project / docs KB).
        Output: ExpectedOutputType.RAG_RESULTS
        """
        q = (query or "").strip()
        deps = depends_on or []

        k = int(top_k) if int(top_k) > 0 else 6

        return ExecutionStep(
            step_id=step_id,
            action=StepAction.USE_RAG_RETRIEVAL,
            enabled=True,
            depends_on=deps,
            budgets=StepBudgets(top_k=k, max_chars=5000, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params={
                # Must match RagRetrievalParams exactly (extra=forbid): query + top_k only.
                "query": q,
                "top_k": k,
            },
            expected_output_type=ExpectedOutputType.RAG_RESULTS,
            rationale_type=RationaleType.RETRIEVE_RAG,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.REPLAN,
                max_retries=0,
                retry_backoff_ms=0,
                replan_reason="RAG retrieval failed",
            ),
        )

    def _step_tools(
        self,
        *,
        tool_input: Dict[str, Any],
        step_id: StepId = StepId.TOOLS,
        depends_on: Optional[List[StepId]] = None,
        max_tool_calls: int = 1,
    ) -> ExecutionStep:
        """
        Execute tool calling step via tools_agent.
        Output: ExpectedOutputType.TOOLS_RESULTS
        """
        deps = depends_on or []

        mtc = int(max_tool_calls) if int(max_tool_calls) > 0 else 1

        return ExecutionStep(
            step_id=step_id,
            action=StepAction.USE_TOOLS,
            enabled=True,
            depends_on=deps,
            budgets=StepBudgets(
                top_k=0,
                max_chars=5000,
                max_tool_calls=mtc,
                max_web_queries=0,
            ),
            inputs={},
            params={
                # Keep schema stable: executor/tools_agent will interpret this payload.
                "input": tool_input or {},
            },
            expected_output_type=ExpectedOutputType.TOOLS_RESULTS,
            rationale_type=RationaleType.RETRIEVE_TOOLS,
            on_failure=FailurePolicy(
                policy=FailurePolicyKind.REPLAN,
                max_retries=0,
                retry_backoff_ms=0,
                replan_reason="Tools execution failed",
            ),
        )


    # -----------------------------
    # Wrapping helpers
    # -----------------------------

    def _wrap(
        self,
        *,
        intent: PlanIntent,
        mode: PlanMode,
        steps: List[ExecutionStep],
        plan_id: Optional[str],
        enforce_finalize: bool = True,
    ) -> ExecutionPlan:
        pid = plan_id or self._new_plan_id()

        # Validate steps count early
        if len(steps) > self._cfg.max_total_steps:
            raise ValueError(
                f"StepPlanner bug: steps_count={len(steps)} exceeds max_total_steps={self._cfg.max_total_steps} "
                f"for intent={intent.value}"
            )

        # Execute plans must have at least one step.
        if mode == PlanMode.EXECUTE and not steps:
            raise ValueError("StepPlanner bug: execute plan has no steps.")

        # Execute plans MUST end with FINALIZE_ANSWER only when we enforce completeness.
        if mode == PlanMode.EXECUTE and enforce_finalize:
            if steps[-1].action != StepAction.FINALIZE_ANSWER:
                raise ValueError(
                    f"StepPlanner bug: execute plan must end with FINALIZE_ANSWER; "
                    f"got last_action={steps[-1].action.value} for intent={intent.value}"
                )

        return ExecutionPlan(
            plan_id=pid,
            intent=intent,
            mode=mode,
            steps=steps,
            budgets=self._plan_budgets(),
            stop_conditions=self._stop_conditions(mode),
            final_answer_style=self._cfg.final_answer_style,
            notes=None,
        )



    # -----------------------------
    # Classification rules (simple + deterministic)
    # -----------------------------

    def _clarifying_question(self, msg: str) -> str:
        return (
            "What exactly should the planner decide or output in your case "
            "(steps/actions/budgets), and what constraints must it follow?"
        )

    def _default_verify_criteria(self, msg: str) -> List[VerifyCriterion]:        
        return [
            VerifyCriterion(id="non_empty", description="Final answer is non-empty", severity=VerifySeverity.ERROR),
            VerifyCriterion(id="no_emojis", description="No emojis in technical output/code", severity=VerifySeverity.WARN),
        ]
    
    def _new_plan_id(self):
        return uuid.uuid4().hex
    
