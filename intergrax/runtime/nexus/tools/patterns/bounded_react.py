# © Artur Czarnecki. All rights reserved.

"""Bounded ReAct tool invocation pattern (TOOL-ENG-18)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.contracts.model_visible_evidence import ModelVisibleEvidenceReference
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.iterative_bounded_tool_loop_policy import (
    reject_sync_iterative_bounded_tool_loop,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.patterns.single_pass import SinglePassPattern
from intergrax.runtime.nexus.tools.tool_invocation_pattern import ToolInvocationResult
from intergrax.runtime.nexus.tools.tool_planner_protocol import ToolPlannerProtocol
from intergrax.tools.core.tool_plan import ToolCallPlan


class BoundedReactPattern:
    """Plan → invoke → observe; multi-round feedback is CE-only (async entry)."""

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
        prior_model_visible_references: Sequence[ModelVisibleEvidenceReference] = (),
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

        reject_sync_iterative_bounded_tool_loop(max_iters)
        raise AssertionError("unreachable after reject_sync_iterative_bounded_tool_loop")
