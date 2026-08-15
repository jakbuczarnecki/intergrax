# © Artur Czarnecki. All rights reserved.

import json

import pytest

from intergrax.contracts.agent_run_trace import GatewayCallStatus, RagCallRecord
from intergrax.contracts.execution_identity import mint_attempt_id, mint_run_id, mint_task_id
from intergrax.contracts.runtime_execution_context import (
    RAG_RETRIEVE_TOOL_ID,
    RuntimeExecutionContext,
    _tool_input_digest,
)
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

_RAW_INPUT = {"source_path": "/tmp/secret-doc.txt", "query": "project X"}


class _Gateway:
    def __init__(self, response: ToolResponse) -> None:
        self._response = response

    async def invoke(self, request: ToolRequest) -> ToolResponse:
        return self._response.model_copy(update={"request_id": request.request_id})


class _EventCollector:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    async def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)


def _exec_ctx(
    *,
    gateway: _Gateway | None = None,
    collector: _EventCollector | None = None,
) -> RuntimeExecutionContext:
    return RuntimeExecutionContext(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        agent_id="local_indexer",
        tool_gateway=gateway,
        event_emitter=collector,
    )


def _tool_request(**overrides: object) -> ToolRequest:
    base = {
        "tool_name": "rag.ingest_document",
        "agent_id": "local_indexer",
        "step_id": "index",
        "input": dict(_RAW_INPUT),
    }
    base.update(overrides)
    return ToolRequest(**base)


def _event_types(events: list[RuntimeEvent]) -> list[RuntimeEventType]:
    return [event.event_type for event in events]


def _assert_payload_safe(payload: dict[str, object]) -> None:
    serialized = json.dumps(payload)
    for raw_value in _RAW_INPUT.values():
        assert raw_value not in serialized
    for key in ("input", "query", "source_path", "text", "content", "chunks"):
        assert key not in payload
    assert "args_digest" in payload
    assert payload["args_digest"] == _tool_input_digest(_RAW_INPUT)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_emits_requested_and_completed_on_success() -> None:
    collector = _EventCollector()
    exec_ctx = _exec_ctx(
        collector=collector,
        gateway=_Gateway(
            ToolResponse(
                request_id="tool-1",
                status=ToolResponseStatus.SUCCESS,
                output={"used": True},
                duration_ms=9,
            )
        ),
    )

    await exec_ctx.invoke_tool(_tool_request(request_id="tool-1"))

    assert _event_types(collector.events) == [
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
    ]
    for event in collector.events:
        _assert_payload_safe(event.payload)
    assert collector.events[1].payload["status"] == "completed"
    assert collector.events[1].payload["latency_ms"] == 9


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_emits_requested_and_denied_when_gateway_missing() -> None:
    collector = _EventCollector()
    exec_ctx = _exec_ctx(collector=collector)

    await exec_ctx.invoke_tool(_tool_request())

    assert _event_types(collector.events) == [
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_DENIED,
    ]
    assert collector.events[1].payload["error_code"] == "tool_gateway_not_configured"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_emits_requested_and_denied_on_denied_response() -> None:
    collector = _EventCollector()
    exec_ctx = _exec_ctx(
        collector=collector,
        gateway=_Gateway(
            ToolResponse(
                request_id="tool-denied",
                status=ToolResponseStatus.DENIED,
                error="tool_not_allowed:rag.ingest_document",
            )
        ),
    )

    await exec_ctx.invoke_tool(_tool_request())

    assert _event_types(collector.events) == [
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_DENIED,
    ]
    assert collector.events[1].payload["error_code"] == "tool_not_allowed:rag.ingest_document"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_emits_requested_and_failed_on_failed_response() -> None:
    collector = _EventCollector()
    exec_ctx = _exec_ctx(
        collector=collector,
        gateway=_Gateway(
            ToolResponse(
                request_id="tool-failed",
                status=ToolResponseStatus.FAILED,
                error="provider_error",
            )
        ),
    )

    await exec_ctx.invoke_tool(_tool_request())

    assert _event_types(collector.events) == [
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_FAILED,
    ]
    assert collector.events[1].payload["status"] == "failed"
    assert collector.events[1].payload["error_code"] == "provider_error"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_event_payload_excludes_raw_args() -> None:
    collector = _EventCollector()
    exec_ctx = _exec_ctx(
        collector=collector,
        gateway=_Gateway(
            ToolResponse(
                request_id="tool-success",
                status=ToolResponseStatus.SUCCESS,
                output={"used": True},
            )
        ),
    )

    await exec_ctx.invoke_tool(_tool_request())

    for event in collector.events:
        _assert_payload_safe(event.payload)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_preserves_tool_and_rag_call_accounting() -> None:
    collector = _EventCollector()
    exec_ctx = _exec_ctx(
        collector=collector,
        gateway=_Gateway(
            ToolResponse(
                request_id="tool-rag-1",
                status=ToolResponseStatus.SUCCESS,
                output={
                    "used": True,
                    "chunks": [{"id": "c1", "text": "hit", "score": 0.9, "metadata": {}}],
                },
                duration_ms=7,
            )
        ),
    )

    await exec_ctx.invoke_tool(
        ToolRequest(
            request_id="tool-rag-1",
            tool_name=RAG_RETRIEVE_TOOL_ID,
            agent_id="local_search",
            step_id="search",
            input={"query": "project X", "workspace_id": "ws-1"},
        )
    )

    tool_calls = exec_ctx.drain_pending_tool_calls()
    rag_calls = exec_ctx.drain_pending_rag_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].status == GatewayCallStatus.SUCCEEDED
    assert len(rag_calls) == 1
    assert isinstance(rag_calls[0], RagCallRecord)
    assert rag_calls[0].hit_count == 1
    assert len(collector.events) == 2
