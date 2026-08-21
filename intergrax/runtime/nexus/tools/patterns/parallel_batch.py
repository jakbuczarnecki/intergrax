# © Artur Czarnecki. All rights reserved.

"""Parallel batch tool invocation pattern (TOOL-ENG-9)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_aggregate import ToolInvocationAggregate
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_loop import execute_planned_tool_calls
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan


class ParallelBatchPattern:
    """Single planner round → parallel read-only invoke → aggregated traces."""

    @property
    def pattern_id(self) -> str:
        return "parallel_batch"

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
            )
            tool_plan = decision.tool_plan

        if tool_plan is None or not tool_plan.calls:
            return ToolInvocationResult(stop_reason="empty_tool_calls")

        max_parallel = max(1, int(state.context.config.max_parallel_tool_calls))
        outcomes = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=tool_plan.calls,
            idempotency_prefix=state.run_id,
            max_parallel_read_only=max_parallel,
        )
        traces = [outcome.trace for outcome in outcomes]
        aggregate = ToolInvocationAggregate.from_traces(traces)
        return ToolInvocationResult(
            tool_traces=list(traces),
            loop_iterations=1,
            stop_reason="legacy_single_pass",
            aggregate=aggregate,
        )
