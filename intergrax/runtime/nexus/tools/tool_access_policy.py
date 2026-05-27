# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional, Sequence

from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class ToolAccessPolicy:
    """
    Enforces agent contract tool permissions before ToolRuntime.invoke (§10.8, Phase B.6).
    """

    @staticmethod
    def apply(
        plan: ToolInvocationPlan,
        *,
        allowed_tools: Optional[Sequence[str]],
        state: Optional[object] = None,
    ) -> ToolInvocationPlan:
        if allowed_tools is None:
            return plan

        allowed = set(allowed_tools)
        use_tools = plan.use_tools and (len(allowed) > 0)

        if plan.use_tools and not use_tools:
            ToolAccessPolicy._emit_denied(
                state,
                message="Tool invocation denied: agent allowed_tools is empty.",
            )

        return ToolInvocationPlan(
            use_rag=plan.use_rag,
            use_websearch=plan.use_websearch,
            use_tools=use_tools,
        )

    @staticmethod
    def is_tool_allowed(tool_name: str, allowed_tools: Optional[Sequence[str]]) -> bool:
        if allowed_tools is None:
            return True
        if not allowed_tools:
            return False
        return tool_name in allowed_tools

    @staticmethod
    def _emit_denied(state: Optional[object], *, message: str) -> None:
        if state is None:
            return
        trace_event = getattr(state, "trace_event", None)
        if callable(trace_event):
            trace_event(
                component=TraceComponent.POLICY,
                step="ToolAccessPolicy",
                message=message,
                level=TraceLevel.WARNING,
            )
