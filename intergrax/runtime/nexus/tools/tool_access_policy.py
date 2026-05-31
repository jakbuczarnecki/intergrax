# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import List, Optional, Sequence

from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.tools.unified.constants import (
    RAG_RETRIEVE_TOOL_ID,
    RAG_TOOL_ALIASES,
    WEBSEARCH_QUERY_TOOL_ID,
    WEBSEARCH_TOOL_ALIASES,
)


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
        normalized = plan.normalized()
        if allowed_tools is None:
            return normalized

        allowed = set(allowed_tools)
        use_rag = normalized.use_rag and ToolAccessPolicy._allows_rag(allowed)
        use_websearch = normalized.use_websearch and ToolAccessPolicy._allows_websearch(allowed)
        use_tools = normalized.use_tools and (len(allowed) > 0)

        filtered_ids = tuple(
            tool_id
            for tool_id in normalized.tool_ids
            if ToolAccessPolicy.is_tool_allowed(tool_id, allowed_tools)
        )

        if normalized.use_rag and not use_rag:
            ToolAccessPolicy._emit_denied(
                state,
                message=f"RAG denied: {RAG_RETRIEVE_TOOL_ID!r} not in allowed_tools.",
            )
        if normalized.use_websearch and not use_websearch:
            ToolAccessPolicy._emit_denied(
                state,
                message=f"Websearch denied: {WEBSEARCH_QUERY_TOOL_ID!r} not in allowed_tools.",
            )
        if normalized.use_tools and not use_tools:
            ToolAccessPolicy._emit_denied(
                state,
                message="Tool invocation denied: agent allowed_tools is empty.",
            )

        return ToolInvocationPlan(
            tool_ids=filtered_ids,
            use_rag=use_rag,
            use_websearch=use_websearch,
            use_tools=use_tools,
        )

    @staticmethod
    def _allows_rag(allowed: set[str]) -> bool:
        return bool(allowed.intersection(RAG_TOOL_ALIASES))

    @staticmethod
    def _allows_websearch(allowed: set[str]) -> bool:
        return bool(allowed.intersection(WEBSEARCH_TOOL_ALIASES))

    @staticmethod
    def is_tool_allowed(tool_name: str, allowed_tools: Optional[Sequence[str]]) -> bool:
        if allowed_tools is None:
            return True
        if not allowed_tools:
            return False
        allowed = set(allowed_tools)
        if tool_name in allowed:
            return True
        if tool_name in RAG_TOOL_ALIASES:
            return ToolAccessPolicy._allows_rag(allowed)
        if tool_name in WEBSEARCH_TOOL_ALIASES:
            return ToolAccessPolicy._allows_websearch(allowed)
        return tool_name in allowed

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
