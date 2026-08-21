# © Artur Czarnecki. All rights reserved.

"""Bounded ReAct tool invocation pattern (TOOL-ENG-18)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.budget.budget_ticks import (
    enforce_wall_time_budget,
    record_planner_iteration_and_enforce,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.investigation_proof import (
    InvestigationProof,
    InvestigationProofStep,
    build_investigation_proof_step,
    collect_available_evidence_ids,
)
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import (
    validate_native_tool_plan_alignment,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.patterns.single_pass import SinglePassPattern
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationResult,
    ToolInvocationStopReason,
)
from intergrax.runtime.nexus.tools.tool_loop import (
    _coerce_messages,
    append_native_tool_messages,
    execute_planned_tool_calls,
    record_identical_tool_call_fingerprints,
    validate_identical_tool_call_repeats,
)
from intergrax.runtime.nexus.tools.tool_planning_policy import tool_choice_for_mode
from intergrax.runtime.nexus.tools.tool_planner_protocol import (
    IterativeToolPlannerProtocol,
    ToolPlannerProtocol,
)
from intergrax.tools.core.tool_plan import ToolCallPlan


class BoundedReactPattern:
    """Plan → invoke → observe loop with native ``role=tool`` messages."""

    @property
    def pattern_id(self) -> str:
        return "bounded_react"

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[ChatMessage],
    ) -> ToolInvocationResult:
        max_iters = max(1, int(max_iterations))
        if max_iters == 1:
            single = SinglePassPattern()
            return single.execute(
                state=state,
                invoker=invoker,
                planner=planner,
                plan=plan,
                allowed_tool_ids=allowed_tool_ids,
                max_iterations=1,
                planner_input=planner_input,
            )

        if not isinstance(planner, IterativeToolPlannerProtocol):
            raise TypeError(
                "Bounded iterative tool invocation (max_iterations > 1) requires "
                "a planner implementing IterativeToolPlannerProtocol"
            )

        messages = _coerce_messages(planner_input)
        appended: list[ChatMessage] = []
        all_traces: list = []
        iterations = 0
        stop_reason: ToolInvocationStopReason = "max_iterations"
        fingerprint_counts: dict[str, int] = {}
        loop_cfg = state.context.config
        proof_steps: list[InvestigationProofStep] = []

        while iterations < max_iters:
            if iterations >= 1:
                enforce_wall_time_budget(state)
            record_planner_iteration_and_enforce(state)
            iterations += 1
            llm_result, tool_plan = planner.plan_native_round(
                messages,
                allowed_tool_ids=allowed_tool_ids,
                run_id=state.run_id,
                tool_choice=tool_choice_for_mode(state.context.config.tools_mode),
            )

            if llm_result.content and not tool_plan.calls:
                stop_reason = "planner_final_answer"
                break

            if not tool_plan.calls:
                stop_reason = "empty_tool_calls"
                break

            validate_native_tool_plan_alignment(llm_result.tool_calls, tool_plan)

            proof_steps.append(
                build_investigation_proof_step(
                    round_index=iterations,
                    assistant_content=llm_result.content,
                    tool_calls=llm_result.tool_calls,
                    messages_before_round=messages,
                )
            )

            validate_identical_tool_call_repeats(
                tool_plan.calls,
                fingerprint_counts=fingerprint_counts,
                max_repeats=loop_cfg.max_identical_tool_call_repeats,
            )
            record_identical_tool_call_fingerprints(tool_plan.calls, fingerprint_counts)
            round_outcomes = execute_planned_tool_calls(
                state=state,
                invoker=invoker,
                calls=tool_plan.calls,
                idempotency_prefix=f"{state.run_id}:loop{iterations}",
            )
            all_traces.extend(outcome.trace for outcome in round_outcomes)
            before = len(messages)
            append_native_tool_messages(
                messages,
                assistant_content=llm_result.content,
                tool_calls=llm_result.tool_calls,
                outcomes=round_outcomes,
            )
            appended.extend(messages[before:])

        investigation_proof: InvestigationProof | None = None
        if proof_steps:
            final_available_evidence_ids: tuple[str, ...] = ()
            if stop_reason == "planner_final_answer":
                final_available_evidence_ids = collect_available_evidence_ids(messages)
            investigation_proof = InvestigationProof(
                steps=tuple(proof_steps),
                final_available_evidence_ids=final_available_evidence_ids,
            )

        return ToolInvocationResult(
            tool_traces=all_traces,
            loop_iterations=iterations,
            stop_reason=stop_reason,
            appended_messages=appended,
            used_native_tool_messages=True,
            investigation_proof=investigation_proof,
        )
