# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.persistence.declarative_tool_executor import CallableDeclarativeToolInvoker
from intergrax.agents.persistence.declarative_tool_executor import DeclarativeToolInvokeResult
from intergrax.agents.persistence.tool_invoker_wiring import (
    attach_declarative_tool_invoker,
    inject_acp_tool_invoker_metadata,
    resolve_declarative_tool_invoker_from_metadata,
    wire_acp_run_request_with_tool_invoker,
)
from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_trace import GatewayCallStatus
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_metadata_roundtrip_for_declarative_invoker() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    metadata = attach_declarative_tool_invoker({}, invoker)
    resolved = resolve_declarative_tool_invoker_from_metadata(metadata)
    assert resolved is invoker


def test_wire_acp_run_request_attaches_invoker() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    request = AgentRunRequest(
        input="x",
        identity=RequestIdentity(tenant_id="t1"),
    )
    wired = wire_acp_run_request_with_tool_invoker(request, invoker)
    assert wired.metadata[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] is invoker


def test_inject_acp_tool_invoker_metadata_wires_host_invoker() -> None:
    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        return DeclarativeToolInvokeResult(status="success")

    invoker = CallableDeclarativeToolInvoker(_invoke)
    metadata: dict[str, object] = {}
    inject_acp_tool_invoker_metadata(
        metadata,
        invoker,
        run_id="run-1",
        agent_id="agent-1",
        tenant_id="tenant-1",
    )
    assert metadata[AcpMetadataKey.DECLARATIVE_TOOL_INVOKER] is invoker


@pytest.mark.asyncio
async def test_runtime_execution_context_records_catalog_tool_calls() -> None:
    class _Gateway:
        async def invoke(self, request: ToolRequest) -> ToolResponse:
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.SUCCESS,
                output={"ok": True},
                duration_ms=7,
            )

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        tool_gateway=_Gateway(),
    )
    response = await exec_ctx.invoke_tool(
        ToolRequest(
            tool_name="rag.ingest_document",
            agent_id="local_indexer",
            step_id="local_indexer_step",
            input={"source_path": "/tmp/doc.txt"},
        )
    )
    assert response.status == ToolResponseStatus.SUCCESS
    pending = exec_ctx.drain_pending_tool_calls()
    assert len(pending) == 1
    assert pending[0].tool_id == "rag.ingest_document"
    assert pending[0].status == GatewayCallStatus.SUCCEEDED
    assert exec_ctx.drain_pending_tool_calls() == []
