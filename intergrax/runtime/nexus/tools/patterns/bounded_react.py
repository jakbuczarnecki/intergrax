# © Artur Czarnecki. All rights reserved.

"""Bounded ReAct tool invocation pattern (TOOL-ENG-18)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
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
)
from intergrax.runtime.nexus.tools.tool_planning_service import ToolPlanningService
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
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
        if max_iters == 1 or not isinstance(planner, ToolPlanningService):
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

        messages = _coerce_messages(planner_input)
        appended: list[ChatMessage] = []
        all_traces: list = []
        iterations = 0
        stop_reason: ToolInvocationStopReason = "max_iterations"

        while iterations < max_iters:
            iterations += 1
            try:
                llm_result, tool_plan = planner.plan_native_round(
                    messages,
                    allowed_tool_ids=allowed_tool_ids,
                    run_id=state.run_id,
                )
            except Exception:
                break

            if llm_result.content and not tool_plan.calls:
                stop_reason = "planner_final_answer"
                break

            if not tool_plan.calls:
                stop_reason = "empty_tool_calls"
                break

            round_traces = execute_planned_tool_calls(
                state=state,
                invoker=invoker,
                calls=tool_plan.calls,
                idempotency_prefix=f"{state.run_id}:loop{iterations}",
            )
            all_traces.extend(round_traces)
            before = len(messages)
            append_native_tool_messages(
                messages,
                assistant_content=llm_result.content,
                tool_calls=llm_result.tool_calls,
                traces=round_traces,
            )
            appended.extend(messages[before:])

        return ToolInvocationResult(
            tool_traces=all_traces,
            loop_iterations=iterations,
            stop_reason=stop_reason,
            appended_messages=appended,
            used_native_tool_messages=True,
        )
