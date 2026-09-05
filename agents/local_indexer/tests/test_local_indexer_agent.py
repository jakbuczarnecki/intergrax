# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolResponse, ToolResponseStatus
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_indexer.contract import build_agent_contract
from local_indexer.steps.index_job import run_index_job
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from testing_support.builder import canonical_execution_identity_scope


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_local_indexer_agent_typed_run_smoke():
    agent = LocalIndexerAgent()
    contract = build_agent_contract()
    with canonical_execution_identity_scope("agent-smoke"):
        result = await agent.run(
            AgentRunRequest(
                input="scaffold smoke",
                identity=RequestIdentity(tenant_id="t1", user_id="u1"),
                agent_id=contract.id,
            )
        )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert "local_indexer" in str(result.output)


class _IngestGateway:
    def __init__(
        self,
        status: ToolResponseStatus,
        *,
        output: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        self.status = status
        self.output = output
        self.error = error

    async def invoke(self, request):
        return ToolResponse(
            request_id=request.request_id,
            status=self.status,
            output=self.output,
            error=self.error,
        )


async def _run_index_job_with_gateway(
    tmp_path,
    monkeypatch,
    gateway: _IngestGateway,
):
    source = tmp_path / "document.txt"
    source.write_text("indexed content", encoding="utf-8")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(tmp_path))
    request = RuntimeRequest(
        agent_id="local_indexer",
        tenant_id="tenant-a",
        user_id="user-a",
        session_id="session-a",
        message="index",
        metadata={"source_paths": [str(source)], "workspace_id": "workspace-a"},
    )
    exec_ctx = RuntimeExecutionContext(
        task_id="task-index",
        run_id="run-index",
        agent_id="local_indexer",
        request=request,
        tool_gateway=gateway,
    )
    step_ctx = AgentStepContext(
        run_id="run-index",
        agent_id="local_indexer",
        metadata={"uaep_exec_ctx": exec_ctx},
    )
    return await run_index_job(step_ctx)


@pytest.mark.asyncio
async def test_local_indexer_treats_domain_noop_as_failed_ingest(
    tmp_path,
    monkeypatch,
):
    result = await _run_index_job_with_gateway(
        tmp_path,
        monkeypatch,
        _IngestGateway(
            ToolResponseStatus.SUCCESS,
            output={
                "used": False,
                "reason": "vectorstore_or_embedding_not_configured",
                "num_chunks": 4,
                "vector_ids": ["must-not-propagate"],
            },
        ),
    )

    summary = result["ingest_summary"]
    assert summary["used"] is False
    assert summary["reason"] == "vectorstore_or_embedding_not_configured"
    assert summary["num_chunks"] == 0
    assert summary["vector_ids"] == []
    assert "ingested=0" in result["summary"]


@pytest.mark.asyncio
async def test_local_indexer_aggregates_semantically_successful_ingest(
    tmp_path,
    monkeypatch,
):
    result = await _run_index_job_with_gateway(
        tmp_path,
        monkeypatch,
        _IngestGateway(
            ToolResponseStatus.SUCCESS,
            output={"used": True, "num_chunks": 2, "vector_ids": ["v1", "v2"]},
        ),
    )

    summary = result["ingest_summary"]
    assert summary["used"] is True
    assert summary["reason"] == "ingest_complete"
    assert summary["num_chunks"] == 2
    assert summary["vector_ids"] == ["v1", "v2"]
    assert "ingested=1" in result["summary"]


@pytest.mark.asyncio
async def test_local_indexer_preserves_transport_failure(
    tmp_path,
    monkeypatch,
):
    result = await _run_index_job_with_gateway(
        tmp_path,
        monkeypatch,
        _IngestGateway(ToolResponseStatus.FAILED, error="tool_transport_failed"),
    )

    summary = result["ingest_summary"]
    assert summary["used"] is False
    assert summary["reason"] == "tool_transport_failed"
    assert summary["num_chunks"] == 0
    assert summary["vector_ids"] == []
    assert "failed=1" in result["summary"]
