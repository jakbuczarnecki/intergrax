# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import uuid
from typing import List, Optional

from intergrax.runtime.nexus.planning.stepplan_models import (
    ExecutionPlan,
    ExecutionStep,
    PlanBudgets,
    PlanIntent,
    PlanMode,
    StepAction,
    StepId,
    StopConditions,
    VerifyCriterion,
    VerifySeverity,
)

from intergrax.runtime.nexus.planning.step_planner.config import StepPlannerConfig
from intergrax.runtime.nexus.planning.step_planner.step_factory import StepPlanStepFactory


class StepPlanAssembly:
    """Plan wrapping, budgets, query builders, and execute-tail chaining."""

    def __init__(self, cfg: StepPlannerConfig, factory: StepPlanStepFactory) -> None:
        self._cfg = cfg
        self._factory = factory

    def plan_budgets(self) -> PlanBudgets:
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


    def stop_conditions(self, mode: PlanMode) -> StopConditions:
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


    def web_query(self, msg: str, *, intent: PlanIntent) -> str:
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


    def ltm_query(self, msg: str, *, intent: PlanIntent) -> str:
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

    def chain_pre_steps(self, steps: List[ExecutionStep]) -> List[ExecutionStep]:
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
    
    def build_execute_tail(
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
            self._factory.synthesize(step_id=StepId.DRAFT, depends_on=depends_on, instructions=msg),
            self._factory.verify(depends_on=[StepId.DRAFT], criteria=self.default_verify_criteria(msg), strict=True),
            self._factory.finalize(depends_on=[StepId.VERIFY], instructions=msg),
        ]
        return steps


    # -----------------------------
    # Plan builders
    # -----------------------------

    def wrap(
        self,
        *,
        intent: PlanIntent,
        mode: PlanMode,
        steps: List[ExecutionStep],
        plan_id: Optional[str],
        enforce_finalize: bool = True,
    ) -> ExecutionPlan:
        pid = plan_id or self.new_plan_id()

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
            budgets=self.plan_budgets(),
            stop_conditions=self.stop_conditions(mode),
            final_answer_style=self._cfg.final_answer_style,
            notes=None,
        )



    # -----------------------------
    # Classification rules (simple + deterministic)
    # -----------------------------

    def clarifying_question(self, msg: str) -> str:
        return (
            "What exactly should the planner decide or output in your case "
            "(steps/actions/budgets), and what constraints must it follow?"
        )

    def default_verify_criteria(self, msg: str) -> List[VerifyCriterion]:        
        return [
            VerifyCriterion(id="non_empty", description="Final answer is non-empty", severity=VerifySeverity.ERROR),
            VerifyCriterion(id="no_emojis", description="No emojis in technical output/code", severity=VerifySeverity.WARN),
        ]
    
    def new_plan_id(self):
        return uuid.uuid4().hex
    
