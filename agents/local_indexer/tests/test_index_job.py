# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from local_indexer.local_indexer_agent import LocalIndexerAgent
from local_indexer.steps.index_job import run_index_job, validate_allowlisted_files


def _step_ctx(
    exec_ctx: RuntimeExecutionContext,
    *,
    run_id: str = "run-test",
) -> AgentStepContext:
    return AgentStepContext(
        run_id=run_id,
        agent_id="local_indexer",
        contract_id="local_indexer",
        metadata={"uaep_exec_ctx": exec_ctx},
    )


@pytest.mark.unit
def test_validate_source_paths_rejects_out_of_scope(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    _, rejected = validate_allowlisted_files([str(outside)], frozenset({str(allowed_root.resolve())}))

    assert rejected == [{"path": str(outside), "reason": "path_not_in_allowlist"}]


@pytest.mark.unit
def test_validate_source_paths_accepts_allowlisted_file(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    doc = allowed_root / "report.txt"
    doc.write_text("hello", encoding="utf-8")

    validated, rejected = validate_allowlisted_files(
        [str(doc)],
        frozenset({str(allowed_root.resolve())}),
    )

    assert rejected == []
    assert validated == [doc.resolve()]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_index_job_ingests_valid_paths(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    doc = allowed_root / "report.txt"
    doc.write_text("hello world", encoding="utf-8")
    original_mtime = doc.stat().st_mtime

    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        assert request.tool_name == RAG_INGEST_TOOL_ID
        assert request.input["source_path"] == str(doc.resolve())
        assert request.input["tenant_id"] == "t1"
        assert request.input["user_id"] == "u1"
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "used": True,
                "num_chunks": 2,
                "vector_ids": ["vec-1", "vec-2"],
                "parser_id": "default",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="index docs",
            metadata={"source_paths": [str(doc)]},
        ),
        tool_gateway=gateway,
    )
    exec_ctx.metadata["runtime_state"] = type(
        "RuntimeStateStub",
        (),
        {
            "context": type(
                "ContextStub",
                (),
                {
                    "config": type(
                        "ConfigStub",
                        (),
                        {"tool_wiring_context": type("WiringStub", (), {"read_allowlist_roots": frozenset({str(allowed_root.resolve())})})()},
                    )()
                },
            )()
        },
    )()

    output = await run_index_job(_step_ctx(exec_ctx))

    assert output["ingest_summary"]["used"] is True
    assert output["ingest_summary"]["num_chunks"] == 2
    assert output["ingest_summary"]["vector_ids"] == ["vec-1", "vec-2"]
    assert output["ingest_summary"]["rejected_paths"] == []
    assert doc.read_text(encoding="utf-8") == "hello world"
    assert doc.stat().st_mtime == original_mtime


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_index_job_fails_safe_without_source_paths() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="index docs",
            metadata={},
        ),
    )

    output = await run_index_job(_step_ctx(exec_ctx))

    assert output["ingest_summary"]["used"] is False
    assert output["ingest_summary"]["reason"] == "source_paths_missing"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_local_indexer_agent_act_returns_ingest_summary(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    doc = allowed_root / "note.txt"
    doc.write_text("content", encoding="utf-8")

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(
        return_value=ToolResponse(
            request_id="tool-1",
            status=ToolResponseStatus.SUCCESS,
            output={"used": True, "num_chunks": 1, "vector_ids": ["vec-1"]},
        )
    )

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="index",
            metadata={"source_paths": [str(doc)]},
        ),
        tool_gateway=gateway,
    )
    exec_ctx.metadata["runtime_state"] = type(
        "RuntimeStateStub",
        (),
        {
            "context": type(
                "ContextStub",
                (),
                {
                    "config": type(
                        "ConfigStub",
                        (),
                        {"tool_wiring_context": type("WiringStub", (), {"read_allowlist_roots": frozenset({str(allowed_root.resolve())})})()},
                    )()
                },
            )()
        },
    )()

    agent = LocalIndexerAgent()
    step_ctx = _step_ctx(exec_ctx)
    observation = await agent.perceive(step_ctx)
    reasoning = await agent.reason(step_ctx, observation)
    output = await agent.act(step_ctx, reasoning)

    assert output["ingest_summary"]["used"] is True
    assert output["ingest_summary"]["num_chunks"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_index_job_preserves_denied_tool_reason(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    doc = allowed_root / "report.txt"
    doc.write_text("hello", encoding="utf-8")

    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.DENIED,
            error="tool_not_allowed:rag.ingest_document",
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="index docs",
            metadata={"source_paths": [str(doc)]},
        ),
        tool_gateway=gateway,
    )
    exec_ctx.metadata["runtime_state"] = type(
        "RuntimeStateStub",
        (),
        {
            "context": type(
                "ContextStub",
                (),
                {
                    "config": type(
                        "ConfigStub",
                        (),
                        {"tool_wiring_context": type("WiringStub", (), {"read_allowlist_roots": frozenset({str(allowed_root.resolve())})})()},
                    )()
                },
            )()
        },
    )()

    output = await run_index_job(_step_ctx(exec_ctx))

    ingested = output["ingest_summary"]["ingested"]
    assert len(ingested) == 1
    assert ingested[0]["status"] == "denied"
    assert ingested[0]["reason"] == "tool_not_allowed:rag.ingest_document"
    assert output["ingest_summary"]["used"] is False
    assert "tool_not_allowed:rag.ingest_document" in str(output["ingest_summary"]["reason"])
    assert "failed=1" in output["answer"]
    assert "tool_error=tool_not_allowed:rag.ingest_document" in output["answer"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_index_job_reads_source_paths_from_step_metadata_without_exec_ctx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    doc = allowed_root / "report.txt"
    doc.write_text("hello", encoding="utf-8")
    monkeypatch.setenv("INTERGRAX_ALLOWED_READ_ROOTS", str(allowed_root.resolve()))

    step_ctx = AgentStepContext(
        run_id="run-acp",
        agent_id="local_indexer",
        contract_id="local_indexer",
        metadata={
            "source_paths": [str(doc)],
            "collection_id": "ws-acp",
        },
    )

    output = await run_index_job(step_ctx)

    assert output["ingest_summary"]["accepted_paths"] == [str(doc.resolve())]
    assert output["ingest_summary"]["rejected_paths"] == [
        {"path": str(doc.resolve()), "reason": "tool_gateway_not_available"}
    ]
    assert output["ingest_summary"]["used"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_index_job_preserves_physical_provenance_without_overrides(
    tmp_path: Path,
) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    doc = allowed_root / "report.txt"
    doc.write_text("hello", encoding="utf-8")
    captured: dict[str, object] = {}

    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        captured["source_path"] = request.input["source_path"]
        captured["metadata"] = request.input["metadata"]
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={"used": True, "num_chunks": 1, "vector_ids": ["v1"]},
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="index docs",
            metadata={"source_paths": [str(doc)]},
        ),
        tool_gateway=gateway,
    )
    exec_ctx.metadata["runtime_state"] = type(
        "RuntimeStateStub",
        (),
        {
            "context": type(
                "ContextStub",
                (),
                {
                    "config": type(
                        "ConfigStub",
                        (),
                        {
                            "tool_wiring_context": type(
                                "WiringStub",
                                (),
                                {"read_allowlist_roots": frozenset({str(allowed_root.resolve())})},
                            )()
                        },
                    )()
                },
            )()
        },
    )()

    output = await run_index_job(_step_ctx(exec_ctx))
    assert captured["source_path"] == str(doc.resolve())
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_path"] == str(doc.resolve())
    assert metadata["file_name"] == "report.txt"
    assert str(allowed_root.resolve()) in str(output["ingest_summary"]["accepted_paths"][0])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_index_job_managed_logical_provenance_override(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    doc = staging / "contract.pdf"
    doc.write_bytes(b"%PDF")
    captured: dict[str, object] = {}

    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        captured["source_path"] = request.input["source_path"]
        captured["metadata"] = request.input["metadata"]
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={"used": True, "num_chunks": 1, "vector_ids": ["v1"]},
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_indexer",
        request=RuntimeRequest(
            agent_id="local_indexer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="index docs",
            metadata={
                "source_paths": [str(doc)],
                "logical_source_path": "managed/src-123/contract.pdf",
                "display_file_name": "contract.pdf",
            },
        ),
        tool_gateway=gateway,
    )
    exec_ctx.metadata["runtime_state"] = type(
        "RuntimeStateStub",
        (),
        {
            "context": type(
                "ContextStub",
                (),
                {
                    "config": type(
                        "ConfigStub",
                        (),
                        {
                            "tool_wiring_context": type(
                                "WiringStub",
                                (),
                                {"read_allowlist_roots": frozenset({str(staging.resolve())})},
                            )()
                        },
                    )()
                },
            )()
        },
    )()

    output = await run_index_job(_step_ctx(exec_ctx))
    assert captured["source_path"] == str(doc.resolve())
    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["source_path"] == "managed/src-123/contract.pdf"
    assert metadata["file_name"] == "contract.pdf"
    summary = output["ingest_summary"]
    assert "managed/src-123/contract.pdf" in summary["accepted_paths"]
    assert str(staging.resolve()) not in str(summary["accepted_paths"])
    assert summary["ingested"][0]["source_path"] == "managed/src-123/contract.pdf"


@pytest.mark.unit
def test_run_index_job_output_attaches_index_summary_diagnostic() -> None:
    output = {
        "ingest_summary": {
            "accepted_paths": ["a.txt"],
            "rejected_paths": [],
            "ingested": [{"status": "success", "num_chunks": 2}],
            "num_chunks": 2,
        }
    }
    from local_indexer.diagnostics import index_diagnostic_from_output

    payload = index_diagnostic_from_output(output)
    assert payload.schema_id() == "lkw.index_summary.v1"
    assert payload.accepted_count == 1
    assert payload.chunk_count == 2
