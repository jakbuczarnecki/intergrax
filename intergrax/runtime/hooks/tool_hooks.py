# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Tool-call hook helpers (§42.20, Appendix B.06)."""

from __future__ import annotations

from typing import Awaitable, Callable, TYPE_CHECKING, Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.hooks.hook_context import HookAction, HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

ToolInvokeFn = Callable[[], Awaitable[ToolResponse]]


def tool_hook_context(
    state: "RuntimeState",
    request: ToolRequest,
    *,
    step_id: Optional[str] = None,
) -> HookContext:
    return HookContext(
        task_id=state.task_id,
        run_id=state.run_id,
        step_id=step_id,
        phase=ExecutionPhase.STEP_EXECUTION,
        runtime_state={
            "tool_id": request.tool_name,
            "tool_name": request.tool_name,
            "request_id": request.request_id,
            "arguments": request.input if isinstance(request.input, dict) else {},
            "capability_ids": [],
            "allowed_tool_ids": [],
        },
    )


def denied_tool_response(request: ToolRequest, *, reason: str) -> ToolResponse:
    return ToolResponse(
        request_id=request.request_id,
        status=ToolResponseStatus.DENIED,
        error=reason,
        duration_ms=0,
    )


async def run_tool_call_hooks(
    middleware: Optional[MiddlewarePipeline],
    ctx: HookContext,
    request: ToolRequest,
    *,
    invoke: ToolInvokeFn,
) -> ToolResponse:
    """Run BEFORE/AFTER_TOOL_CALL around ``invoke``."""
    if middleware is None:
        return await invoke()

    before = await middleware.run_before(HookPoint.BEFORE_TOOL_CALL, ctx)
    if before.action == HookAction.BLOCK:
        return denied_tool_response(
            request,
            reason=before.reason or "tool_call_blocked_by_hook",
        )

    try:
        return await invoke()
    finally:
        await middleware.run_after(HookPoint.AFTER_TOOL_CALL, ctx)
