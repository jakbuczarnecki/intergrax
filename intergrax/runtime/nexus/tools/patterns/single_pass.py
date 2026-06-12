# © Artur Czarnecki. All rights reserved.

"""Single-pass tool invocation pattern (TOOL-ENG-17)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_loop import execute_planned_tool_calls
from intergrax.runtime.nexus.tools.tool_planning_policy import tool_choice_for_mode
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan


class SinglePassPattern:
    """One planner round → sequential invoke (legacy ``max_tool_iterations == 1``)."""

    @property
    def pattern_id(self) -> str:
        return "single_pass"

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
        _ = max_iterations
        tool_plan = plan
        if tool_plan is None:
            decision = planner.plan_tools(
                input_data=planner_input,
                context=None,
                run_id=state.run_id,
                allowed_tool_ids=allowed_tool_ids,
                tool_choice=tool_choice_for_mode(state.context.config.tools_mode),
            )
            tool_plan = decision.tool_plan

        if tool_plan is None or not tool_plan.calls:
            return ToolInvocationResult(stop_reason="empty_tool_calls")

        traces = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=tool_plan.calls,
            idempotency_prefix=state.run_id,
        )
        return ToolInvocationResult(
            tool_traces=traces,
            loop_iterations=1,
            stop_reason="legacy_single_pass",
        )
