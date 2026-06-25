# © Artur Czarnecki. All rights reserved.

"""LKW.1.4 — contract-level acceptance: index → search → synthesize."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from intergrax.tools.providers.rag.service import RAG_TOOL_ID
from intergrax.tools.providers.workspace.service import WORKSPACE_WRITE_FILE_TOOL_ID
from local_indexer.steps.index_job import run_index_job
from local_search.steps.search_job import run_search_job
from local_synthesizer.steps.synthesize_job import run_synthesize_job

_FIXTURE_TEXT = "Intergrax LKW acceptance fixture — searchable paragraph."
_QUERY = "Intergrax LKW acceptance"
_COLLECTION_ID = "lkw-acceptance-ws"


def _runtime_state_stub(allowed_root: Path) -> object:
    return type(
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


def _step_ctx(
    exec_ctx: RuntimeExecutionContext,
    *,
    agent_id: str,
    run_id: str,
    message: str = "",
    extra_metadata: dict[str, object] | None = None,
) -> AgentStepContext:
    metadata: dict[str, object] = {"uaep_exec_ctx": exec_ctx}
    if extra_metadata:
        metadata.update(extra_metadata)
    return AgentStepContext(
        run_id=run_id,
        agent_id=agent_id,
        contract_id=agent_id,
        message=message,
        metadata=metadata,
    )


def _exec_ctx(
    *,
    agent_id: str,
    run_id: str,
    message: str,
    metadata: dict[str, object],
    gateway: AsyncMock,
    runtime_state: object,
) -> RuntimeExecutionContext:
    exec_ctx = RuntimeExecutionContext(
        task_id=f"task-{run_id}",
        run_id=run_id,
        agent_id=agent_id,
        request=RuntimeRequest(
            agent_id=agent_id,
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message=message,
            metadata=metadata,
        ),
        tool_gateway=gateway,
    )
    exec_ctx.metadata["runtime_state"] = runtime_state
    return exec_ctx


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lkw_acceptance_index_search_synthesize_contract_flow(tmp_path: Path) -> None:
    allowed_root = tmp_path / "workspace"
    allowed_root.mkdir()
    fixture_doc = allowed_root / "fixture.txt"
    fixture_doc.write_text(_FIXTURE_TEXT, encoding="utf-8")
    original_content = fixture_doc.read_text(encoding="utf-8")
    original_mtime = fixture_doc.stat().st_mtime
    fixture_path = str(fixture_doc.resolve())

    ingest_calls: list[ToolRequest] = []
    retrieve_calls: list[ToolRequest] = []
    write_calls: list[ToolRequest] = []

    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        if request.tool_name == RAG_INGEST_TOOL_ID:
            ingest_calls.append(request)
            assert request.input["source_path"] == fixture_path
            assert request.input["workspace_id"] == _COLLECTION_ID
            assert request.input["metadata"]["collection_id"] == _COLLECTION_ID
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.SUCCESS,
                output={
                    "used": True,
                    "status": "success",
                    "num_chunks": 1,
                    "vector_ids": ["vec-fixture-1"],
                    "parser_id": "default",
                },
            )
        if request.tool_name == RAG_TOOL_ID:
            retrieve_calls.append(request)
            assert request.input["query"] == _QUERY
            assert request.input["workspace_id"] == _COLLECTION_ID
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.SUCCESS,
                output={
                    "used": True,
                    "chunks": [
                        {
                            "id": "chunk-fixture-1",
                            "text": _FIXTURE_TEXT,
                            "score": 0.95,
                            "metadata": {"source_path": fixture_path},
                        }
                    ],
                    "citations": [],
                    "context_text": _FIXTURE_TEXT,
                    "reason": "",
                },
            )
        if request.tool_name == WORKSPACE_WRITE_FILE_TOOL_ID:
            write_calls.append(request)
            assert _FIXTURE_TEXT in request.input["content"]
            assert fixture_path in request.input["content"]
            return ToolResponse(
                request_id=request.request_id,
                status=ToolResponseStatus.SUCCESS,
                output={
                    "artifact_id": "art-shadow-1",
                    "relative_path": "synthesis-draft.md",
                    "size_bytes": len(request.input["content"]),
                    "content_type": "text/markdown",
                    "sha256": "deadbeef",
                    "workspace_id": "shadow-ws-1",
                },
            )
        raise AssertionError(f"unexpected tool: {request.tool_name}")

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)
    runtime_state = _runtime_state_stub(allowed_root)

    index_exec = _exec_ctx(
        agent_id="local_indexer",
        run_id="run-index",
        message="index fixture",
        metadata={"source_paths": [fixture_path], "collection_id": _COLLECTION_ID},
        gateway=gateway,
        runtime_state=runtime_state,
    )
    index_output = await run_index_job(_step_ctx(index_exec, agent_id="local_indexer", run_id="run-index"))

    assert index_output["ingest_summary"]["used"] is True
    assert index_output["ingest_summary"]["reason"] == "ingest_complete"
    assert len(ingest_calls) == 1
    assert ingest_calls[0].tool_name == RAG_INGEST_TOOL_ID

    search_exec = _exec_ctx(
        agent_id="local_search",
        run_id="run-search",
        message=_QUERY,
        metadata={"query": _QUERY, "collection_id": _COLLECTION_ID, "top_k": 5},
        gateway=gateway,
        runtime_state=runtime_state,
    )
    search_output = await run_search_job(_step_ctx(search_exec, agent_id="local_search", run_id="run-search"))

    search_summary = search_output["search_summary"]
    assert search_summary["used"] is True
    assert search_summary["reason"] == "retrieve_complete"
    assert len(retrieve_calls) == 1
    assert retrieve_calls[0].tool_name == RAG_TOOL_ID
    assert search_summary["num_results"] == 1
    evidence = search_summary["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["text"] == _FIXTURE_TEXT
    assert evidence[0]["source_path"] == fixture_path
    assert evidence[0]["chunk_id"] == "chunk-fixture-1"

    synthesize_exec = _exec_ctx(
        agent_id="local_synthesizer",
        run_id="run-synthesize",
        message="synthesize draft",
        metadata={
            "shadow_workspace": True,
            "output_name": "synthesis-draft.md",
            "search_summary": search_summary,
            "evidence": evidence,
        },
        gateway=gateway,
        runtime_state=runtime_state,
    )
    synthesize_output = await run_synthesize_job(
        _step_ctx(synthesize_exec, agent_id="local_synthesizer", run_id="run-synthesize")
    )

    synth_summary = synthesize_output["synthesize_summary"]
    assert synth_summary["used"] is True
    assert synth_summary["reason"] == "write_complete"
    assert synth_summary["shadow_workspace"] is True
    assert len(write_calls) == 1
    assert write_calls[0].tool_name == WORKSPACE_WRITE_FILE_TOOL_ID

    assert fixture_doc.read_text(encoding="utf-8") == original_content
    assert fixture_doc.stat().st_mtime == original_mtime
