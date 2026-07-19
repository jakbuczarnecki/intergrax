# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import (
    ToolRequest,
    ToolResponse,
    ToolResponseStatus,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID
from local_search.diagnostics import SearchSummaryReason
from local_search.steps.search_job import run_search_job


def _step_ctx(
    exec_ctx: RuntimeExecutionContext | None,
    *,
    run_id: str = "run-test",
    message: str = "",
) -> AgentStepContext:
    metadata: dict[str, object] = {}
    if exec_ctx is not None:
        metadata["uaep_exec_ctx"] = exec_ctx
    return AgentStepContext(
        run_id=run_id,
        agent_id="local_search",
        contract_id="local_search",
        message=message,
        metadata=metadata,
    )


def _search_summary(output: dict[str, object]) -> dict[str, Any]:
    summary = output["search_summary"]
    assert isinstance(summary, dict)
    return cast(dict[str, Any], summary)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_search_job_fails_safe_without_query() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_search",
        request=RuntimeRequest(
            agent_id="local_search",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="",
            metadata={},
        ),
    )

    summary = _search_summary(await run_search_job(_step_ctx(exec_ctx)))

    assert summary["used"] is False
    assert summary["reason"] == SearchSummaryReason.QUERY_MISSING.value
    assert summary["evidence"] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_search_job_fails_safe_without_tool_gateway() -> None:
    summary = _search_summary(
        await run_search_job(_step_ctx(None, message="find docs about X"))
    )

    assert summary["used"] is False
    assert summary["reason"] == (SearchSummaryReason.TOOL_GATEWAY_NOT_AVAILABLE.value)
    assert summary["query"] == "find docs about X"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_search_job_retrieves_with_valid_query() -> None:
    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        assert request.tool_name == RAG_RETRIEVE_TOOL_ID
        assert request.input["query"] == "project X"
        assert request.input["top_k"] == 3
        assert request.input["workspace_id"] == "ws-1"
        assert request.input["tenant_id"] == "t1"
        assert request.input["user_id"] == "u1"
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "used": True,
                "chunks": [
                    {
                        "id": "chunk-1",
                        "text": "Project X overview",
                        "score": 0.91,
                        "metadata": {"source_path": "/data/report.txt"},
                    }
                ],
                "citations": [],
                "context_text": "Project X overview",
                "reason": "",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_search",
        request=RuntimeRequest(
            agent_id="local_search",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="ignored when metadata query set",
            metadata={"query": "project X", "collection_id": "ws-1", "top_k": 3},
        ),
        tool_gateway=gateway,
    )

    summary = _search_summary(await run_search_job(_step_ctx(exec_ctx)))

    assert summary["used"] is True
    assert summary["reason"] == SearchSummaryReason.RETRIEVE_COMPLETE.value
    assert summary["query"] == "project X"
    assert summary["collection_id"] == "ws-1"
    assert summary["num_results"] == 1
    assert summary["evidence"] == [
        {
            "text": "Project X overview",
            "source_path": "/data/report.txt",
            "chunk_id": "chunk-1",
            "score": 0.91,
        }
    ]

    rag_calls = exec_ctx.drain_pending_rag_calls()
    assert len(rag_calls) == 1
    assert rag_calls[0].collection_id == "ws-1"
    assert rag_calls[0].hit_count == 1
    assert len(exec_ctx.drain_pending_tool_calls()) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_search_job_fails_safe_on_retrieve_error() -> None:
    gateway = AsyncMock()
    gateway.invoke = AsyncMock(
        return_value=ToolResponse(
            request_id="tool-1",
            status=ToolResponseStatus.FAILED,
            error="vectorstore_unavailable",
        )
    )

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_search",
        request=RuntimeRequest(
            agent_id="local_search",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="find something",
            metadata={},
        ),
        tool_gateway=gateway,
    )

    summary = _search_summary(
        await run_search_job(_step_ctx(exec_ctx, message="find something"))
    )

    assert summary["used"] is False
    assert summary["reason"] == SearchSummaryReason.RETRIEVE_FAILED.value


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_search_job_preserves_raw_tool_reason() -> None:
    gateway = AsyncMock()
    gateway.invoke = AsyncMock(
        return_value=ToolResponse(
            request_id="tool-1",
            status=ToolResponseStatus.SUCCESS,
            output={"used": False, "reason": "retriever_failed"},
        )
    )

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_search",
        request=RuntimeRequest(
            agent_id="local_search",
            tenant_id="lkw-smoke",
            user_id="u1",
            session_id="s1",
            message="find marker",
            metadata={"query": "marker", "collection_id": "ws-1"},
        ),
        tool_gateway=gateway,
    )

    summary = _search_summary(
        await run_search_job(_step_ctx(exec_ctx, message="find marker"))
    )

    assert summary["reason"] == SearchSummaryReason.RETRIEVE_FAILED.value
    assert summary["raw_tool_reason"] == "retriever_failed"


@pytest.mark.unit
def test_run_search_job_output_attaches_search_summary_diagnostic() -> None:
    output: dict[str, object] = {
        "search_summary": {
            "query": "find docs",
            "num_results": 1,
            "evidence": [{"source_path": "docs/a.md"}],
            "raw_tool_reason": "retriever_failed",
        }
    }
    from local_search.diagnostics import search_diagnostic_from_output

    payload = search_diagnostic_from_output(output)
    assert payload.schema_id() == "lkw.search_summary.v1"
    assert payload.num_results == 1
    assert payload.evidence_count == 1
    assert payload.raw_tool_reason == "retriever_failed"
