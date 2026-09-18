# © Artur Czarnecki. All rights reserved.

"""Reference ToolInvocationPattern surface for the enterprise multi-capability package."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.tools.core.tool_plan import ToolCallPlan
from intergrax.tools.invocation_pattern.contracts import (
    ToolInvocationInvokerPort,
    ToolInvocationPatternContext,
    ToolInvocationPatternResult,
    ToolInvocationPlannerPort,
)


class ReferenceEnterpriseSinglePassPattern:
    """Deterministic offline pattern — returns empty tool calls without network."""

    @property
    def pattern_id(self) -> str:
        return "reference_enterprise_single_pass"

    def execute(
        self,
        *,
        context: ToolInvocationPatternContext,
        invoker: ToolInvocationInvokerPort,
        planner: ToolInvocationPlannerPort,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationPatternResult:
        _ = context, invoker, planner, plan, allowed_tool_ids, max_iterations, planner_input
        return ToolInvocationPatternResult(
            pattern_id="reference_enterprise_single_pass",
            stop_reason="empty_tool_calls",
        )
