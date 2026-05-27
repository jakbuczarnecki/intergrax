# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""§42.12 tool gateway — ``ToolRequest`` / ``ToolResponse`` facade over ``ToolRuntime``."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional, Sequence

from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
from intergrax.runtime.nexus.tools.tool_runtime import ToolInvocationPlan, ToolRuntime

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

NEXUS_CAPABILITY_PLAN = "nexus.capability_plan"
NEXUS_RAG = "nexus.rag"
NEXUS_WEBSEARCH = "nexus.websearch"
NEXUS_TOOLS = "nexus.tools"

_KNOWN_CAPABILITY_TOOLS = frozenset(
    {NEXUS_CAPABILITY_PLAN, NEXUS_RAG, NEXUS_WEBSEARCH, NEXUS_TOOLS, "rag", "websearch", "tools"}
)


class RuntimeToolGateway:
    """
    Tier-1 tool gateway (§42.12).

    Agents MUST invoke capabilities through ``ToolRequest`` — not direct Nexus step imports.
    """

    def __init__(
        self,
        *,
        state: "RuntimeState",
        allowed_tools: Optional[Sequence[str]] = None,
        trace_step: str = "ToolGateway",
    ) -> None:
        self._state = state
        self._allowed_tools = list(allowed_tools) if allowed_tools is not None else None
        self._trace_step = trace_step

    @classmethod
    def for_state(
        cls,
        state: "RuntimeState",
        *,
        allowed_tools: Optional[Sequence[str]] = None,
        trace_step: str = "ToolGateway",
    ) -> "RuntimeToolGateway":
        return cls(state=state, allowed_tools=allowed_tools, trace_step=trace_step)

    async def invoke(self, request: ToolRequest) -> ToolResponse:
        started = time.perf_counter()
        if request.tool_name not in _KNOWN_CAPABILITY_TOOLS:
            if not ToolAccessPolicy.is_tool_allowed(request.tool_name, self._allowed_tools):
                return ToolResponse(
                    request_id=request.request_id,
                    status=ToolResponseStatus.DENIED,
                    error=f"tool_not_allowed:{request.tool_name}",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.FAILED,
                error=f"unknown_capability_tool:{request.tool_name}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        plan = self._plan_from_request(request)
        try:
            result = await ToolRuntime.invoke(
                state=self._state,
                plan=plan,
                trace_step=self._trace_step,
                allowed_tools=self._allowed_tools,
            )
        except Exception as exc:  # noqa: BLE001 — gateway boundary
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.FAILED,
                error=str(exc),
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "used_rag": result.used_rag,
                "used_websearch": result.used_websearch,
                "used_tools": result.used_tools,
                "tool_trace_count": result.tool_trace_count,
            },
            duration_ms=duration_ms,
            trace_ref=self._state.run_id or "",
        )

    @staticmethod
    def _plan_from_request(request: ToolRequest) -> ToolInvocationPlan:
        name = request.tool_name
        payload = request.input

        if name in {NEXUS_CAPABILITY_PLAN, "capability_plan"}:
            return ToolInvocationPlan(
                use_rag=bool(payload.get("use_rag", False)),
                use_websearch=bool(payload.get("use_websearch", False)),
                use_tools=bool(payload.get("use_tools", False)),
            )
        if name in {NEXUS_RAG, "rag"}:
            return ToolInvocationPlan(use_rag=True)
        if name in {NEXUS_WEBSEARCH, "websearch"}:
            return ToolInvocationPlan(use_websearch=True)
        if name in {NEXUS_TOOLS, "tools"}:
            return ToolInvocationPlan(use_tools=True)
        return ToolInvocationPlan()
