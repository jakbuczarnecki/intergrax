# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intergrax.runtime.nexus.planning.stepplan_models import (
    ExecutionStep,
    ExpectedOutputType,
    FailurePolicy,
    FailurePolicyKind,
    OutputFormat,
    RationaleType,
    StepAction,
    StepBudgets,
    StepId,
    VerifyCriterion,
    VerifySeverity,
    WebSearchStrategy,
)

from intergrax.runtime.nexus.planning.step_planner.config import StepPlannerConfig


class StepPlanStepFactory:
    """Deterministic ExecutionStep builders (params are always dicts)."""

    def __init__(self, cfg: StepPlannerConfig) -> None:
        self._cfg = cfg

    def synthesize(self, *, step_id: StepId, depends_on: List[StepId], instructions: str) -> ExecutionStep:
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

    def verify(
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

    def finalize(self, *, depends_on: List[StepId], instructions: str) -> ExecutionStep:
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

    def websearch(self, *, step_id: StepId, depends_on: List[StepId], query: str) -> ExecutionStep:
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

    def ltm(self, *, step_id: StepId, depends_on: List[StepId], query: str) -> ExecutionStep:
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

    def clarify(self, *, step_id: StepId, depends_on: List[StepId], question: str) -> ExecutionStep:
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
                "context_key": "clarify.user_input",
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
    
    def rag_retrieval(
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

    def tools(
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


