# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""UAEP helper — tool gateway bound to ``RuntimeState`` in step metadata."""

from __future__ import annotations

from typing import Optional, Sequence

from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.middleware.pipeline import MiddlewarePipeline
from intergrax.runtime.nexus.tools.tool_gateway import RuntimeToolGateway
from intergrax.runtime.nexus.tools.uaep_invocation_wiring import UAEPToolInvocationWiringResolver
from intergrax.tools.invocation_wiring import ToolInvocationContext


class BoundToolGateway:
    """
    Resolves ``RuntimeState`` from UAEP ``exec_ctx.metadata['runtime_state']``.

    All physical tool calls route through ``RuntimeToolGateway`` and ``RuntimeToolInvoker``.
    Per-invocation dependencies are projected via ``UAEPToolInvocationWiringResolver``.
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
        self._wiring_resolver = UAEPToolInvocationWiringResolver(exec_ctx)

    async def invoke(self, request: ToolRequest) -> ToolResponse:
        state = self._exec_ctx.metadata.get("runtime_state")
        if state is None:
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.DENIED,
                error="runtime_state_not_bound",
            )
        invocation_context = ToolInvocationContext(
            run_id=state.run_id,
            step_id=request.step_id or self._trace_step,
            tool_id=request.tool_name,
            agent_id=request.agent_id,
            tenant_id=getattr(state, "tenant_id", None),
            correlation_request_id=request.request_id,
            wiring_resolver=self._wiring_resolver,
        )
        gateway = RuntimeToolGateway.for_state(
            state,
            allowed_tools=self._allowed_tools,
            trace_step=self._trace_step,
            middleware=self._middleware,
        )
        return await gateway.invoke(request, invocation_context=invocation_context)
