# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from intergrax.runtime.drop_in_knowledge_mode.planning.stepplan_models import (
    # core models
    ExecutionPlan,
    ExecutionStep,
    StepAction,
    StepBudgets,
    PlanBudgets,
    StopConditions,
    FailurePolicy,
    VerifyCriterion,
    # enums
    WebSearchStrategy,
    OutputFormat,
    PlanMode,
    # params models (names must match stepplan_models.py)
    AskClarifyingParams,
    LtmSearchParams,
    AttachmentsRetrievalParams,
    RagRetrievalParams,
    WebSearchParams,
    ToolsParams,
    SynthesizeDraftParams,
    VerifyAnswerParams,
    FinalizeAnswerParams,
)


@dataclass(frozen=True)
class StepPlannerConfig:
    # Plan-level defaults
    max_total_steps: int = 6
    max_total_tool_calls: int = 10
    max_total_web_queries: int = 5
    max_total_chars_context: int = 12000
    max_total_tokens_output: Optional[int] = None

    # Step-level defaults
    step_max_chars: int = 2000
    step_top_k: int = 5

    # Websearch
    web_recency_days: int = 30
    web_max_results: int = 5
    web_strategy: WebSearchStrategy = WebSearchStrategy.HYBRID

    # Verification defaults
    verify_strict: bool = True

    # Deterministic pruning priority if we exceed max_total_steps:
    # remove in this order (lowest value first)
    # tools -> web -> rag -> attachments -> ltm
    prune_order: Tuple[str, ...] = ("tools", "web", "rag", "attachments", "ltm")


class StepPlanner:
    """
    Deterministic (rule-based) step planner.

    - No LLM usage.
    - Produces a stable ExecutionPlan compatible with stepplan_models.py.
    - Retrieval steps are always before SYNTHESIZE_DRAFT.
    - Execute plans always end with FINALIZE_ANSWER.
    """

    def __init__(self, config: Optional[StepPlannerConfig] = None):
        self._cfg = config or StepPlannerConfig()

    def plan(
        self,
        user_message: str,
        *,
        # engine-level routing/gating (simple booleans)
        enable_websearch: bool = False,
        enable_rag: bool = False,
        enable_tools: bool = False,
        enable_attachments: bool = False,
        enable_user_ltm: bool = False,
        # optional hints
        require_code_example: bool = False,
        final_format: OutputFormat = OutputFormat.MARKDOWN,
        plan_id: str = "stepplan-001",
    ) -> ExecutionPlan:
        msg = (user_message or "").strip()
        if not msg:
            return self._clarify("Empty user message. What should the assistant do?", plan_id=plan_id)

        intent = self._make_intent(msg)

        # Clarify ONLY if truly ambiguous (rare)
        if self._is_ambiguous(msg):
            return self._clarify(self._clarifying_question(msg), intent=intent, plan_id=plan_id)

        # Build steps in deterministic order
        steps: List[ExecutionStep] = []

        # Retrieval (deterministic ordering)
        if enable_user_ltm:
            steps.append(self._step_user_ltm(query=self._ltm_query(msg)))

        if enable_attachments:
            steps.append(self._step_attachments(query=self._attachments_query(msg)))

        if enable_rag:
            steps.append(self._step_rag(query=self._rag_query(msg)))

        if enable_websearch:
            steps.append(self._step_websearch(query=self._web_query(msg)))

        if enable_tools:
            steps.append(self._step_tools(instructions="Use tools if needed."))

        # Core execution tail
        steps.append(
            self._step_synthesize(
                instructions=self._draft_instructions(msg),
                must_include=self._must_include(msg),
                avoid=[],
            )
        )
        steps.append(
            self._step_verify(
                criteria=self._verify_criteria(require_code_example=require_code_example),
                strict=self._cfg.verify_strict,
            )
        )
        steps.append(
            self._step_finalize(
                instructions=self._final_instructions(msg),
                fmt=final_format,
            )
        )

        # Ensure depends_on chain is correct + stay within max_total_steps
        steps = self._chain_and_prune(steps, max_steps=self._cfg.max_total_steps)

        plan = ExecutionPlan(
            plan_id=plan_id,
            intent=intent,
            mode=PlanMode.EXECUTE,
            steps=steps,
            budgets=PlanBudgets(
                max_total_steps=self._cfg.max_total_steps,
                max_total_tool_calls=self._cfg.max_total_tool_calls,
                max_total_web_queries=self._cfg.max_total_web_queries,
                max_total_chars_context=self._cfg.max_total_chars_context,
                max_total_tokens_output=self._cfg.max_total_tokens_output,
            ),
            stop_conditions=StopConditions(
                stop_on_clarifying_question_answered=True,
                stop_on_verifier_pass=True,
                stop_on_budget_exhausted=True,
            ),
            final_answer_style="concise_technical",
            notes=None,
        )
        return plan

    # -----------------------------
    # Step factories (return ExecutionStep)
    # NOTE: ExecutionStep.params is a dict; we pass model_dump() to match schema exactly.
    # -----------------------------

    def _step_user_ltm(self, *, query: str) -> ExecutionStep:
        return ExecutionStep(
            step_id="ltm",
            action=StepAction.USE_USER_LONGTERM_MEMORY_SEARCH,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(
                top_k=self._cfg.step_top_k,
                max_chars=self._cfg.step_max_chars,
                max_tool_calls=0,
                max_web_queries=0,
            ),
            inputs={},
            params=LtmSearchParams(
                query=query,
                top_k=self._cfg.step_top_k,
                score_threshold=None,
                include_debug=False,
            ).model_dump(),
            expected_output="User long-term memory hits",
            rationale="Gather project/user context deterministically",
            on_failure=FailurePolicy(policy="skip", max_retries=0, retry_backoff_ms=0, replan_reason=None),
        )

    def _step_attachments(self, *, query: str) -> ExecutionStep:
        return ExecutionStep(
            step_id="attachments",
            action=StepAction.USE_ATTACHMENTS_RETRIEVAL,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(
                top_k=self._cfg.step_top_k,
                max_chars=self._cfg.step_max_chars,
                max_tool_calls=0,
                max_web_queries=0,
            ),
            inputs={},
            params=AttachmentsRetrievalParams(query=query, top_k=self._cfg.step_top_k).model_dump(),
            expected_output="Attachment retrieval hits",
            rationale="Use session attachments as context",
            on_failure=FailurePolicy(policy="skip", max_retries=0, retry_backoff_ms=0, replan_reason=None),
        )

    def _step_rag(self, *, query: str) -> ExecutionStep:
        return ExecutionStep(
            step_id="rag",
            action=StepAction.USE_RAG_RETRIEVAL,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(
                top_k=self._cfg.step_top_k,
                max_chars=self._cfg.step_max_chars,
                max_tool_calls=0,
                max_web_queries=0,
            ),
            inputs={},
            params=RagRetrievalParams(query=query, top_k=self._cfg.step_top_k).model_dump(),
            expected_output="RAG hits",
            rationale="Retrieve from configured RAG store",
            on_failure=FailurePolicy(policy="skip", max_retries=0, retry_backoff_ms=0, replan_reason=None),
        )

    def _step_websearch(self, *, query: str) -> ExecutionStep:
        return ExecutionStep(
            step_id="web",
            action=StepAction.USE_WEBSEARCH,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(
                top_k=self._cfg.web_max_results,
                max_chars=self._cfg.step_max_chars,
                max_tool_calls=0,
                max_web_queries=1,
            ),
            inputs={},
            params=WebSearchParams(
                query=query,
                recency_days=self._cfg.web_recency_days,
                max_results=self._cfg.web_max_results,
                strategy=self._cfg.web_strategy,
                domains_allowlist=None,
            ).model_dump(),
            expected_output="Web search results",
            rationale="Freshness / external verification needed",
            on_failure=FailurePolicy(policy="replan", max_retries=0, retry_backoff_ms=0, replan_reason="Websearch failed"),
        )

    def _step_tools(self, *, instructions: str) -> ExecutionStep:
        # stepplan_models.ToolsParams has only: input: Dict[str, Any]
        return ExecutionStep(
            step_id="tools",
            action=StepAction.USE_TOOLS,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(
                top_k=0,
                max_chars=self._cfg.step_max_chars,
                max_tool_calls=1,
                max_web_queries=0,
            ),
            inputs={},
            params=ToolsParams(input={"instructions": instructions}).model_dump(),
            expected_output="Tool outputs (if any)",
            rationale="Optional tool usage",
            on_failure=FailurePolicy(policy="skip", max_retries=0, retry_backoff_ms=0, replan_reason=None),
        )

    def _step_synthesize(
        self,
        *,
        instructions: str,
        must_include: List[str],
        avoid: List[str],
    ) -> ExecutionStep:
        return ExecutionStep(
            step_id="draft",
            action=StepAction.SYNTHESIZE_DRAFT,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(
                top_k=0,
                max_chars=self._cfg.step_max_chars,
                max_tool_calls=0,
                max_web_queries=0,
            ),
            inputs={},
            params=SynthesizeDraftParams(instructions=instructions, must_include=must_include, avoid=avoid).model_dump(),
            expected_output="Draft answer text",
            rationale="Generate a draft from gathered context",
            on_failure=FailurePolicy(policy="retry", max_retries=1, retry_backoff_ms=0, replan_reason=None),
        )

    def _step_verify(self, *, criteria: List[VerifyCriterion], strict: bool) -> ExecutionStep:
        # Schema requires at least one criterion; enforce deterministically
        if not criteria:
            criteria = [VerifyCriterion(id="non_empty", description="Answer is non-empty", severity="error")]

        return ExecutionStep(
            step_id="verify",
            action=StepAction.VERIFY_ANSWER,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(top_k=0, max_chars=1000, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params=VerifyAnswerParams(criteria=criteria, strict=strict).model_dump(),
            expected_output="Verification pass/fail",
            rationale="Guardrail check before final answer",
            on_failure=FailurePolicy(policy="replan", max_retries=0, retry_backoff_ms=0, replan_reason="Verification failed"),
        )

    def _step_finalize(self, *, instructions: str, fmt: OutputFormat) -> ExecutionStep:
        return ExecutionStep(
            step_id="final",
            action=StepAction.FINALIZE_ANSWER,
            enabled=True,
            depends_on=[],
            budgets=StepBudgets(top_k=0, max_chars=self._cfg.step_max_chars, max_tool_calls=0, max_web_queries=0),
            inputs={},
            params=FinalizeAnswerParams(instructions=instructions, format=fmt).model_dump(),
            expected_output="Final assistant answer",
            rationale="Finalize format and style",
            on_failure=FailurePolicy(policy="fail", max_retries=0, retry_backoff_ms=0, replan_reason=None),
        )

    # -----------------------------
    # Clarify plan
    # -----------------------------

    def _clarify(self, question: str, *, intent: str = "clarify", plan_id: str = "stepplan-001") -> ExecutionPlan:
        steps = [
            ExecutionStep(
                step_id="clarify",
                action=StepAction.ASK_CLARIFYING_QUESTION,
                enabled=True,
                depends_on=[],
                budgets=StepBudgets(top_k=0, max_chars=200, max_tool_calls=0, max_web_queries=0),
                inputs={},
                params=AskClarifyingParams(question=question).model_dump(),
                expected_output="User clarification",
                rationale="Ambiguous request",
                on_failure=FailurePolicy(policy="fail", max_retries=0, retry_backoff_ms=0, replan_reason=None),
            )
        ]
        return ExecutionPlan(
            plan_id=plan_id,
            intent=intent,
            mode=PlanMode.CLARIFY,
            steps=steps,
            budgets=PlanBudgets(
                max_total_steps=1,
                max_total_tool_calls=0,
                max_total_web_queries=0,
                max_total_chars_context=self._cfg.max_total_chars_context,
                max_total_tokens_output=self._cfg.max_total_tokens_output,
            ),
            stop_conditions=StopConditions(
                stop_on_clarifying_question_answered=True,
                stop_on_verifier_pass=False,
                stop_on_budget_exhausted=True,
            ),
            final_answer_style="concise_technical",
            notes=None,
        )

    # -----------------------------
    # Deterministic chaining + pruning
    # -----------------------------

    def _chain_and_prune(self, steps: List[ExecutionStep], *, max_steps: int) -> List[ExecutionStep]:
        """
        Ensures depends_on is a simple linear chain AND prunes optional retrieval steps
        if we exceed max_steps (deterministic order).
        """
        # Prune if needed
        if max_steps and len(steps) > max_steps:
            steps = self._prune_to_budget(steps, max_steps=max_steps)

        # Re-chain depends_on linearly (no forward refs)
        prev_id: Optional[str] = None
        for s in steps:
            s.depends_on = [] if prev_id is None else [prev_id]
            prev_id = s.step_id

        return steps

    def _prune_to_budget(self, steps: List[ExecutionStep], *, max_steps: int) -> List[ExecutionStep]:
        """
        Drops optional steps deterministically to fit the budget.
        Never drops: draft/verify/final (must exist in execute mode).
        """
        core_ids = {"draft", "verify", "final"}
        core = [s for s in steps if s.step_id in core_ids]
        optional = [s for s in steps if s.step_id not in core_ids]

        # If even core is too big (shouldn't happen), truncate from the left (but keep final step)
        if len(core) > max_steps:
            # keep last max_steps steps
            core = core[-max_steps:]
            return core

        # We can keep at most (max_steps - len(core)) optional steps
        keep_optional = max_steps - len(core)
        if keep_optional <= 0:
            return core

        # Deterministic keep order: follow current build order, but prune by configured priority
        # We remove by prune_order first, until it fits.
        opt_by_id = {s.step_id: s for s in optional}
        opt_ids_in_order = [s.step_id for s in optional]

        # Remove until fits
        while len(opt_ids_in_order) > keep_optional:
            removed = False
            for candidate in self._cfg.prune_order:
                if candidate in opt_by_id and candidate in opt_ids_in_order:
                    opt_ids_in_order.remove(candidate)
                    removed = True
                    break
            if not removed:
                # fallback: remove oldest optional
                opt_ids_in_order.pop(0)

        kept_optional_steps = [opt_by_id[sid] for sid in opt_ids_in_order if sid in opt_by_id]

        # Preserve original overall order
        kept_ids = set(s.step_id for s in core) | set(s.step_id for s in kept_optional_steps)
        out = [s for s in steps if s.step_id in kept_ids]
        return out

    # -----------------------------
    # Rules / heuristics (deterministic)
    # -----------------------------

    def _make_intent(self, msg: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9]+", "_", msg.strip().lower())
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:64] or "intent"

    def _is_ambiguous(self, msg: str) -> bool:
        # Conservative: clarify only when truly ambiguous
        short = len(msg) < 12
        vague = bool(re.fullmatch(r"(help|assist|explain|what about this)\.?\s*", msg.strip().lower()))
        return short or vague

    def _clarifying_question(self, msg: str) -> str:
        return "What exactly should the assistant produce (e.g., code, explanation, or a step-by-step plan), and what constraints apply?"

    # Query builders
    def _web_query(self, msg: str) -> str:
        return msg

    def _rag_query(self, msg: str) -> str:
        return msg

    def _ltm_query(self, msg: str) -> str:
        return msg

    def _attachments_query(self, msg: str) -> str:
        return "Find relevant information in session attachments for: " + msg

    # Draft/final instructions
    def _draft_instructions(self, msg: str) -> str:
        return msg

    def _final_instructions(self, msg: str) -> str:
        return msg

    def _must_include(self, msg: str) -> List[str]:
        return []

    def _verify_criteria(self, *, require_code_example: bool) -> List[VerifyCriterion]:
        crit = [VerifyCriterion(id="non_empty", description="Answer is non-empty", severity="error")]
        if require_code_example:
            crit.append(VerifyCriterion(id="has_code", description="Includes a code example", severity="error"))
        return crit
