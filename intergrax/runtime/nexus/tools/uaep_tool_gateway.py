# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UAEP helper — tool gateway bound to ``RuntimeState`` in step metadata."""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway


class BoundToolGateway:
    """
    Resolves ``RuntimeState`` from UAEP ``exec_ctx.metadata['runtime_state']``.

    Agents set ``runtime_state`` during ``run_step`` before calling ``ctx.invoke_tool``.
    """

    def __init__(
        self,
        exec_ctx: RuntimeExecutionContext,
        *,
        allowed_tools: Optional[Sequence[str]] = None,
        trace_step: str = "UAEPToolGateway",
    ) -> None:
        self._exec_ctx = exec_ctx
        self._allowed_tools = allowed_tools
        self._trace_step = trace_step

    async def invoke(self, request: ToolRequest) -> ToolResponse:
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
        )
        return await gateway.invoke(request)
