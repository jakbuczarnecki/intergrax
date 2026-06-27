# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_run_trace import GatewayCallStatus, RagCallRecord
from intergrax.contracts.runtime_execution_context import (
    RAG_RETRIEVE_TOOL_ID,
    RuntimeExecutionContext,
    build_rag_call_record,
)
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus


class _Gateway:
    def __init__(self, response: ToolResponse) -> None:
        self._response = response

    async def invoke(self, request: ToolRequest) -> ToolResponse:
        return self._response.model_copy(update={"request_id": request.request_id})


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_records_pending_rag_call_for_rag_retrieve() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_search",
        tool_gateway=_Gateway(
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

    response = await exec_ctx.invoke_tool(
        ToolRequest(
            request_id="tool-rag-1",
            tool_name=RAG_RETRIEVE_TOOL_ID,
            agent_id="local_search",
            step_id="search",
            input={"query": "project X", "workspace_id": "ws-1"},
        )
    )
    _ = response

    rag_calls = exec_ctx.drain_pending_rag_calls()
    assert len(rag_calls) == 1
    assert isinstance(rag_calls[0], RagCallRecord)
    assert rag_calls[0].call_id == "tool-rag-1"
    assert rag_calls[0].collection_id == "ws-1"
    assert rag_calls[0].status == GatewayCallStatus.SUCCEEDED
    assert rag_calls[0].latency_ms == 7
    assert rag_calls[0].hit_count == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_does_not_record_rag_call_for_non_rag_tools() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="demo",
        tool_gateway=_Gateway(
            ToolResponse(
                request_id="tool-fs-1",
                status=ToolResponseStatus.SUCCESS,
                output={"content": "ok"},
                duration_ms=3,
            )
        ),
    )

    await exec_ctx.invoke_tool(
        ToolRequest(
            tool_name="filesystem.read",
            agent_id="demo",
            step_id="read",
            input={"path": "/tmp/x.txt"},
        )
    )

    assert exec_ctx.drain_pending_rag_calls() == []
    assert len(exec_ctx.drain_pending_tool_calls()) == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_invoke_tool_does_not_record_rag_call_for_rag_ingest_document() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        tool_gateway=_Gateway(
            ToolResponse(
                request_id="tool-ingest-1",
                status=ToolResponseStatus.SUCCESS,
                output={"used": True, "num_chunks": 4},
                duration_ms=11,
            )
        ),
    )

    await exec_ctx.invoke_tool(
        ToolRequest(
            tool_name="rag.ingest_document",
            agent_id="local_indexer",
            step_id="index",
            input={"source_path": "/tmp/doc.txt"},
        )
    )

    assert exec_ctx.drain_pending_rag_calls() == []
    tool_calls = exec_ctx.drain_pending_tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_id == "rag.ingest_document"


@pytest.mark.unit
@pytest.mark.gate
def test_build_rag_call_record_prefers_explicit_hit_count_and_collection_id() -> None:
    record = build_rag_call_record(
        call_id="tool-1",
        tool_id=RAG_RETRIEVE_TOOL_ID,
        tool_input={"collection_id": "col-a", "query": "x"},
        status=GatewayCallStatus.SUCCEEDED,
        latency_ms=5,
        output={"hit_count": 2, "chunks": [{"id": "only-one"}]},
    )

    assert record is not None
    assert record.collection_id == "col-a"
    assert record.hit_count == 2
