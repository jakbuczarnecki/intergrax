# © Artur Czarnecki. All rights reserved.

"""Parallel semantic batch pattern (TOOL-ENG-25)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config_types import ToolSelectionMode
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_aggregate import ToolInvocationAggregate
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_input_defaults import default_tool_input
from intergrax.runtime.nexus.tools.tool_loop import execute_planned_tool_calls
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.runtime.nexus.tools.tool_selection import (
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids,
)
from intergrax.tools.core.tool_plan import PlannedToolCall, ToolCallPlan


class ParallelSemanticBatchPattern:
    """Semantic top-k selection → parallel read-only invoke → aggregate."""

    @property
    def pattern_id(self) -> str:
        return "parallel_semantic_batch"

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
        _ = planner, plan, max_iterations
        registry = invoker.registry
        query = _planner_query_text(planner_input)
        selected = resolve_planner_allowed_tool_ids(
            ToolSelectionMode.SEMANTIC,
            ToolSelectionContext(
                registry=registry,
                query=query,
                skill_profile=state.context.config.skill_profile,
                plan_allowed_tool_ids=allowed_tool_ids,
                top_k=state.context.config.tool_selection_top_k,
                embedding_manager=state.context.config.embedding_manager,
            ),
        )
        if not selected:
            return ToolInvocationResult(stop_reason="empty_tool_calls")

        calls: list[PlannedToolCall] = []
        for index, tool_id in enumerate(selected):
            contract = registry.get(tool_id).contract
            tool_input = default_tool_input(contract, query)
            if tool_input is None:
                continue
            calls.append(
                PlannedToolCall(
                    step_id=f"semantic-{index}",
                    tool_id=tool_id,
                    input=tool_input,
                )
            )
        if not calls:
            return ToolInvocationResult(stop_reason="empty_tool_calls")

        max_parallel = max(1, int(state.context.config.max_parallel_tool_calls))
        outcomes = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=calls,
            idempotency_prefix=f"{state.run_id}:semantic",
            max_parallel_read_only=max_parallel,
        )
        traces = [outcome.trace for outcome in outcomes]
        aggregate = ToolInvocationAggregate.from_traces(traces)
        return ToolInvocationResult(
            tool_traces=traces,
            loop_iterations=1,
            stop_reason="legacy_single_pass",
            aggregate=aggregate,
        )


def _planner_query_text(planner_input: str | list[ChatMessage]) -> str:
    if isinstance(planner_input, str):
        return planner_input
    for message in reversed(planner_input):
        if message.role == "user" and message.content:
            return message.content
    return ""
