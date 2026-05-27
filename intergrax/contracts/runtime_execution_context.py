# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified per-run execution context (architecture §42.13.1)."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.tool_request import ToolRequest, ToolResponse
from intergrax.runtime.events.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent


@runtime_checkable
class ToolGateway(Protocol):
    async def invoke(self, request: ToolRequest) -> ToolResponse: ...


@runtime_checkable
class EventEmitter(Protocol):
    async def emit(self, event: RuntimeEvent) -> None: ...


@runtime_checkable
class MemoryView(Protocol):
    async def read(self, namespace: str, key: str) -> Optional[Dict[str, Any]]: ...

    async def write(self, namespace: str, key: str, value: Dict[str, Any]) -> None: ...


@runtime_checkable
class TraceWriter(Protocol):
    def write(self, label: str, payload: Dict[str, Any]) -> None: ...


class RuntimeExecutionContext(BaseModel):
    """
    Context passed to agent ``run_step`` under UAEP (§42.5).

    Agents receive this — never raw adapter clients or global singletons.
    """

    task_id: str
    run_id: str
    node_id: Optional[str] = None
    agent_id: str
    correlation_id: str = ""
    phase: ExecutionPhase = ExecutionPhase.STEP_EXECUTION
    contract: Optional[AgentContract] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}

    # Protocol-typed gateways — set by AgentEngine at runtime (not serialized)
    tool_gateway: Optional[Any] = Field(default=None, exclude=True)
    event_emitter: Optional[Any] = Field(default=None, exclude=True)
    memory_view: Optional[Any] = Field(default=None, exclude=True)
    trace: Optional[Any] = Field(default=None, exclude=True)
    request: Optional[Any] = Field(default=None, exclude=True)
    domain_context: Optional[Any] = Field(default=None, exclude=True)

    async def emit_event(self, event: RuntimeEvent) -> None:
        if self.event_emitter is not None:
            await self.event_emitter.emit(event)

    async def invoke_tool(self, request: ToolRequest) -> ToolResponse:
        if self.tool_gateway is None:
            from intergrax.contracts.tool_request import ToolResponseStatus

            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.DENIED,
                error="tool_gateway_not_configured",
            )
        return await self.tool_gateway.invoke(request)
