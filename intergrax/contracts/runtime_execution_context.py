# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unified per-run execution context (architecture §42.13.1)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from pydantic import BaseModel, Field

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run_trace import GatewayCallStatus, RagCallRecord, ToolCallRecord
from intergrax.contracts.memory_write_policy import MemoryWritePolicy
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.contracts.execution_phase import ExecutionPhase

_PENDING_TOOL_CALLS_KEY = "_pending_tool_call_records"
_PENDING_RAG_CALLS_KEY = "_pending_rag_call_records"
RAG_RETRIEVE_TOOL_ID = "rag.retrieve"
RAG_INGEST_TOOL_ID = "rag.ingest_document"
WORKSPACE_WRITE_FILE_TOOL_ID = "workspace.write_file"

if TYPE_CHECKING:
    from intergrax.runtime.events.runtime_event import RuntimeEvent


@runtime_checkable
class MetadataCarrier(Protocol):
    metadata: Dict[str, Any]


@runtime_checkable
class ToolGateway(Protocol):
    async def invoke(self, request: ToolRequest) -> ToolResponse: ...


@runtime_checkable
class EventEmitter(Protocol):
    async def emit(self, event: RuntimeEvent) -> None: ...


@runtime_checkable
class MemoryView(Protocol):
    async def read(self, namespace: str, key: str) -> Optional[Dict[str, Any]]: ...

    async def write(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
        *,
        policy: MemoryWritePolicy = MemoryWritePolicy.REPLACE,
    ) -> None: ...

    async def list(self, namespace: str, prefix: str = "") -> List[Any]: ...


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

    tool_gateway: Optional[Any] = Field(default=None, exclude=True)
    event_emitter: Optional[Any] = Field(default=None, exclude=True)
    memory_view: Optional[Any] = Field(default=None, exclude=True)
    trace: Optional[Any] = Field(default=None, exclude=True)
    request: Optional[MetadataCarrier] = Field(default=None, exclude=True)
    domain_context: Optional[Any] = Field(default=None, exclude=True)

    async def emit_event(self, event: RuntimeEvent) -> None:
        if self.event_emitter is not None:
            await self.event_emitter.emit(event)

    async def invoke_tool(self, request: ToolRequest) -> ToolResponse:
        tool_input = dict(request.input or {})
        args_digest = _tool_input_digest(tool_input)
        await self._emit_immediate_tool_event(
            _tool_requested_event_type(),
            request=request,
            args_digest=args_digest,
            status="requested",
        )
        started = time.perf_counter()
        if self.tool_gateway is None:
            response = ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.DENIED,
                error="tool_gateway_not_configured",
            )
        else:
            response = await self.tool_gateway.invoke(request)
        latency_ms = (
            response.duration_ms
            if response.duration_ms > 0
            else int((time.perf_counter() - started) * 1000)
        )
        self._record_tool_call(request, response, latency_ms=latency_ms)
        await self._emit_immediate_tool_event(
            _immediate_tool_outcome_event_type(response.status),
            request=request,
            args_digest=args_digest,
            status=_immediate_tool_outcome_status(response.status),
            latency_ms=latency_ms,
            error_code=response.error if response.status != ToolResponseStatus.SUCCESS else None,
        )
        return response

    async def _emit_immediate_tool_event(
        self,
        event_type: Any,
        *,
        request: ToolRequest,
        args_digest: str,
        status: str,
        latency_ms: int | None = None,
        error_code: str | None = None,
    ) -> None:
        from intergrax.runtime.events.runtime_event import RuntimeEvent

        payload: dict[str, Any] = {
            "tool_id": request.tool_name,
            "status": status,
            "args_digest": args_digest,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "phase": self.phase.value,
        }
        if latency_ms is not None:
            payload["latency_ms"] = latency_ms
        if error_code:
            payload["error_code"] = error_code
        event = RuntimeEvent(
            event_type=event_type,
            phase=self.phase,
            payload=payload,
            agent_id=self.agent_id,
            task_id=self.task_id,
            run_id=self.run_id,
            node_id=self.node_id,
            correlation_id=self.correlation_id,
            step_id=request.step_id or None,
        )
        await self.emit_event(event)

    def drain_pending_tool_calls(self) -> list[ToolCallRecord]:
        raw = self.metadata.pop(_PENDING_TOOL_CALLS_KEY, None)
        if not raw:
            return []
        return list(raw)

    def drain_pending_rag_calls(self) -> list[RagCallRecord]:
        raw = self.metadata.pop(_PENDING_RAG_CALLS_KEY, None)
        if not raw:
            return []
        return list(raw)

    def _record_tool_call(
        self,
        request: ToolRequest,
        response: ToolResponse,
        *,
        latency_ms: int,
    ) -> None:
        if response.status == ToolResponseStatus.SUCCESS:
            status = GatewayCallStatus.SUCCEEDED
        elif response.status == ToolResponseStatus.DENIED:
            status = GatewayCallStatus.DENIED
        else:
            status = GatewayCallStatus.FAILED

        call_id = response.request_id or request.request_id
        tool_input = dict(request.input or {})
        pending: list[ToolCallRecord] = list(self.metadata.get(_PENDING_TOOL_CALLS_KEY) or [])
        pending.append(
            ToolCallRecord(
                call_id=call_id,
                tool_id=request.tool_name,
                status=status,
                latency_ms=latency_ms,
                args_digest=_tool_input_digest(tool_input),
                error_code=response.error if status != GatewayCallStatus.SUCCEEDED else None,
            )
        )
        self.metadata[_PENDING_TOOL_CALLS_KEY] = pending

        rag_record = build_rag_call_record(
            call_id=call_id,
            tool_id=request.tool_name,
            tool_input=tool_input,
            status=status,
            latency_ms=latency_ms,
            output=response.output,
        )
        if rag_record is not None:
            pending_rag: list[RagCallRecord] = list(self.metadata.get(_PENDING_RAG_CALLS_KEY) or [])
            pending_rag.append(rag_record)
            self.metadata[_PENDING_RAG_CALLS_KEY] = pending_rag

    def should_cancel(self) -> bool:
        from intergrax.runtime.cancellation.coordinator import CancellationCoordinator

        if isinstance(self.request, MetadataCarrier):
            if CancellationCoordinator.is_requested(self.request.metadata):
                return True
        return CancellationCoordinator.is_requested(self.metadata)


def _tool_input_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _tool_requested_event_type() -> Any:
    from intergrax.runtime.events.runtime_event import RuntimeEventType

    return RuntimeEventType.TOOL_REQUESTED


def _immediate_tool_outcome_event_type(status: ToolResponseStatus) -> Any:
    from intergrax.runtime.events.runtime_event import RuntimeEventType

    if status == ToolResponseStatus.SUCCESS:
        return RuntimeEventType.TOOL_COMPLETED
    if status == ToolResponseStatus.DENIED:
        return RuntimeEventType.TOOL_DENIED
    return RuntimeEventType.TOOL_FAILED


def _immediate_tool_outcome_status(status: ToolResponseStatus) -> str:
    if status == ToolResponseStatus.SUCCESS:
        return "completed"
    if status == ToolResponseStatus.DENIED:
        return "denied"
    return "failed"


def is_rag_retrieve_tool(tool_id: str) -> bool:
    return tool_id == RAG_RETRIEVE_TOOL_ID


def build_rag_call_record(
    *,
    call_id: str,
    tool_id: str,
    tool_input: dict[str, Any],
    status: GatewayCallStatus,
    latency_ms: int,
    output: dict[str, Any] | None,
    policy_rule_id: str | None = None,
) -> RagCallRecord | None:
    if not is_rag_retrieve_tool(tool_id):
        return None
    return RagCallRecord(
        call_id=call_id,
        collection_id=_resolve_rag_collection_id(tool_input),
        status=status,
        latency_ms=latency_ms,
        hit_count=_resolve_rag_retrieve_hit_count(output),
        policy_rule_id=policy_rule_id,
    )


def _resolve_rag_collection_id(tool_input: dict[str, Any]) -> str:
    for key in ("collection_id", "workspace_id"):
        raw = tool_input.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return ""


def _resolve_rag_retrieve_hit_count(output: dict[str, Any] | None) -> int:
    if not output:
        return 0
    for key in ("hit_count", "num_results"):
        if key in output:
            try:
                return max(0, int(output[key]))
            except (TypeError, ValueError):
                return 0
    chunks = output.get("chunks")
    if isinstance(chunks, list):
        return len(chunks)
    return 0
