# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UAEP helper — tool gateway bound to ``RuntimeState`` in step metadata."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
from intergrax.runtime.nexus.tools.runtime_bound_catalog import (
    invoke_runtime_bound_tool,
    is_runtime_bound_tool,
)
from intergrax.runtime.sandbox.sandbox_runtime import SANDBOX_TOOL_NAME


class BoundToolGateway:
    """
    Resolves ``RuntimeState`` from UAEP ``exec_ctx.metadata['runtime_state']``.

    Agents set ``runtime_state`` during ``run_step`` before calling ``ctx.invoke_tool``.
    Sandbox tools route through ``exec_ctx.metadata['sandbox_session']`` (§42.12.2).
    """

    def __init__(
        self,
        exec_ctx: RuntimeExecutionContext,
        *,
        allowed_tools: Optional[Sequence[str]] = None,
        trace_step: str = "UAEPToolGateway",
        middleware: Optional[MiddlewarePipeline] = None,
    ) -> None:
        self._exec_ctx = exec_ctx
        self._allowed_tools = allowed_tools
        self._trace_step = trace_step
        self._middleware = middleware

    async def invoke(self, request: ToolRequest) -> ToolResponse:
        if request.tool_name == SANDBOX_TOOL_NAME:
            return await self._invoke_sandbox(request)
        if is_runtime_bound_tool(request.tool_name):
            return invoke_runtime_bound_tool(self._exec_ctx, request)

        state = self._exec_ctx.metadata.get("runtime_state")
        if state is None:
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.DENIED,
                error="runtime_state_not_bound",
            )
        gateway = RuntimeToolGateway.for_state(
            state,
            allowed_tools=self._allowed_tools,
            trace_step=self._trace_step,
            middleware=self._middleware,
        )
        return await gateway.invoke(request)

    async def _invoke_sandbox(self, request: ToolRequest) -> ToolResponse:
        started = time.perf_counter()
        if not ToolAccessPolicy.is_tool_allowed(SANDBOX_TOOL_NAME, self._allowed_tools):
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.DENIED,
                error=f"tool_not_allowed:{SANDBOX_TOOL_NAME}",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        session = self._exec_ctx.metadata.get("sandbox_session")
        if session is None:
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.DENIED,
                error="sandbox_not_configured",
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        operation = str(request.input.get("operation", ""))
        payload = dict(request.input.get("payload") or {})
        result = session.execute(operation, payload)
        duration_ms = int((time.perf_counter() - started) * 1000)

        if not result.success:
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.FAILED,
                error=result.error or "sandbox_execution_failed",
                duration_ms=duration_ms,
            )

        output = dict(result.output)
        if result.audit_entry is not None:
            output["audit_entry_id"] = result.audit_entry.entry_id
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output=output,
            duration_ms=duration_ms,
            trace_ref=session.session_id,
        )
