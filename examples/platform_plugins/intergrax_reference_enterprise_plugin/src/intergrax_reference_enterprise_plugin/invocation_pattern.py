# © Artur Czarnecki. All rights reserved.

"""Reference ToolInvocationPattern surface for the enterprise multi-capability package."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan


class ReferenceEnterpriseSinglePassPattern:
    """Deterministic offline pattern — returns empty tool calls without network."""

    @property
    def pattern_id(self) -> str:
        return "reference_enterprise_single_pass"

    def execute(
        self,
        *,
        state: RuntimeState,
        invoker: RuntimeToolInvoker,
        planner: ToolPlannerProtocol,
        plan: ToolCallPlan | None,
        allowed_tool_ids: Sequence[str] | None,
        max_iterations: int,
        planner_input: str | list[object],
    ) -> ToolInvocationResult:
        _ = state, invoker, planner, plan, allowed_tool_ids, max_iterations, planner_input
        return ToolInvocationResult(
            pattern_id="reference_enterprise_single_pass",
            stop_reason="empty_tool_calls",
        )
