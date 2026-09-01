# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.tools.providers.workspace.service import WORKSPACE_WRITE_FILE_TOOL_ID
from local_synthesizer.steps.synthesize_job import run_synthesize_job


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
        agent_id="local_synthesizer",
        contract_id="local_synthesizer",
        message=message,
        metadata=metadata,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_rejects_non_shadow_workspace() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="prepare report",
            metadata={"draft": "hello", "shadow_workspace": False},
        ),
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is False
    assert summary["reason"] == "shadow_workspace_required"
    assert summary["shadow_workspace"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_consumes_prior_search_handoff() -> None:
    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        assert request.tool_name == WORKSPACE_WRITE_FILE_TOOL_ID
        assert "Pipeline evidence paragraph" in request.input["content"]
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "artifact_id": "art-pipeline",
                "relative_path": "pipeline-draft.md",
                "workspace_id": "shadow-ws-pipeline",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-pipeline",
        run_id="run-pipeline",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="synthesize pipeline draft",
            metadata={
                "shadow_workspace": True,
                "output_name": "pipeline-draft.md",
                "prior_agent_outputs": {
                    "node_local_search": {
                        "agent_id": "local_search",
                        "summary": "local_search: search job — query='pipeline', results=1",
                        "structured_data": {
                            "search_summary": {
                                "used": True,
                                "reason": "retrieve_complete",
                                "query": "pipeline",
                                "num_results": 1,
                                "evidence": [
                                    {
                                        "text": "Pipeline evidence paragraph",
                                        "source_path": "/data/fixture.txt",
                                        "chunk_id": "chunk-pipeline-1",
                                    }
                                ],
                                "selected_artifact_ref": "chunk:chunk-pipeline-1",
                            }
                        },
                    }
                },
            },
        ),
        tool_gateway=gateway,
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is True
    assert summary["reason"] == "write_complete"
    assert summary["num_evidence_items"] == 1
    assert summary["artifact_path"] == "pipeline-draft.md"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_prefers_explicit_evidence_over_prior_handoff() -> None:
    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        assert "Explicit evidence wins" in request.input["content"]
        assert "Prior evidence ignored" not in request.input["content"]
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "artifact_id": "art-explicit",
                "relative_path": "explicit-draft.md",
                "workspace_id": "shadow-ws-explicit",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-explicit",
        run_id="run-explicit",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="synthesize",
            metadata={
                "shadow_workspace": True,
                "output_name": "explicit-draft.md",
                "evidence": [{"text": "Explicit evidence wins", "source_path": "/explicit.txt"}],
                "prior_agent_outputs": {
                    "node_local_search": {
                        "agent_id": "local_search",
                        "structured_data": {
                            "search_summary": {
                                "evidence": [{"text": "Prior evidence ignored"}],
                            }
                        },
                    }
                },
            },
        ),
        tool_gateway=gateway,
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is True
    assert summary["num_evidence_items"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_reads_shared_context_reads_handoff() -> None:
    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        assert "Shared context evidence" in request.input["content"]
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "artifact_id": "art-shared",
                "relative_path": "shared-draft.md",
                "workspace_id": "shadow-ws-shared",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-shared",
        run_id="run-shared",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="synthesize",
            metadata={
                "shadow_workspace": True,
                "shared_context_reads": {
                    "node_local_search": {
                        "agent_id": "local_search",
                        "structured_data": {
                            "search_summary": {
                                "query": "shared query",
                                "evidence": [
                                    {
                                        "text": "Shared context evidence",
                                        "source_path": "/shared.txt",
                                    }
                                ],
                            }
                        },
                    }
                },
            },
        ),
        tool_gateway=gateway,
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is True
    assert summary["reason"] == "write_complete"
    assert summary["num_evidence_items"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_fails_safe_without_content() -> None:
    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="",
            metadata={"shadow_workspace": True},
        ),
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is False
    assert summary["reason"] == "content_missing"
    assert summary["shadow_workspace"] is True
    assert summary["num_evidence_items"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_fails_safe_without_tool_gateway() -> None:
    output = await run_synthesize_job(
        AgentStepContext(
            run_id="run-test",
            agent_id="local_synthesizer",
            contract_id="local_synthesizer",
            message="prepare client email",
            metadata={"shadow_workspace": True, "draft": "email body"},
        )
    )

    summary = output["synthesize_summary"]
    assert summary["used"] is False
    assert summary["reason"] == "tool_gateway_not_available"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_writes_draft_to_shadow_workspace() -> None:
    async def _invoke_tool(request: ToolRequest) -> ToolResponse:
        assert request.tool_name == WORKSPACE_WRITE_FILE_TOOL_ID
        assert request.input["path"] == "client-email.md"
        assert "Project X overview" in request.input["content"]
        assert request.input["content_type"] == "text/markdown"
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
            output={
                "artifact_id": "art-1",
                "relative_path": "client-email.md",
                "size_bytes": 42,
                "content_type": "text/markdown",
                "sha256": "abc",
                "workspace_id": "shadow-ws-1",
            },
        )

    gateway = AsyncMock()
    gateway.invoke = AsyncMock(side_effect=_invoke_tool)

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="prepare client email",
            metadata={
                "shadow_workspace": True,
                "output_name": "client-email.md",
                "search_summary": {"query": "project X"},
                "selected_artifact_ref": "chunk:chunk-1",
                "evidence": [
                    {
                        "text": "Project X overview",
                        "source_path": "/data/report.txt",
                        "chunk_id": "chunk-1",
                    }
                ],
            },
        ),
        tool_gateway=gateway,
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is True
    assert summary["reason"] == "write_complete"
    assert summary["output_name"] == "client-email.md"
    assert summary["artifact_path"] == "client-email.md"
    assert summary["artifact_ref"] == "shadow-ws-1/art-1"
    assert summary["shadow_workspace"] is True
    assert summary["num_evidence_items"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_synthesize_job_fails_safe_on_write_error() -> None:
    gateway = AsyncMock()
    gateway.invoke = AsyncMock(
        return_value=ToolResponse(
            request_id="tool-1",
            status=ToolResponseStatus.FAILED,
            error="shadow_workspace_not_configured",
        )
    )

    exec_ctx = RuntimeExecutionContext(
        task_id="task-1",
        run_id="run-1",
        agent_id="local_synthesizer",
        request=RuntimeRequest(
            agent_id="local_synthesizer",
            tenant_id="t1",
            user_id="u1",
            session_id="s1",
            message="ignored",
            metadata={"shadow_workspace": True, "draft": "final body"},
        ),
        tool_gateway=gateway,
    )

    output = await run_synthesize_job(_step_ctx(exec_ctx))

    summary = output["synthesize_summary"]
    assert summary["used"] is False
    assert summary["reason"] == "write_failed"
    assert summary["shadow_workspace"] is True
    assert summary["raw_tool_reason"] == "shadow_workspace_not_configured"


@pytest.mark.unit
def test_run_synthesize_job_output_attaches_synthesize_summary_diagnostic() -> None:
    output = {
        "synthesize_summary": {
            "used": True,
            "reason": "write_complete",
            "shadow_workspace": True,
            "num_evidence_items": 2,
            "artifact_path": "draft.md",
            "artifact_ref": "shadow-ws-1/art-1",
        }
    }
    from local_synthesizer.diagnostics import synthesize_diagnostic_from_output

    payload = synthesize_diagnostic_from_output(output)
    assert payload.schema_id() == "lkw.synthesize_summary.v1"
    assert payload.write_status == "write_complete"
    assert payload.shadow_write is True
    assert payload.source_evidence_count == 2
    assert payload.artifact_path == "draft.md"
    assert payload.artifact_ref == "shadow-ws-1/art-1"
    redacted = payload.redact().to_dict()
    assert "content" not in redacted
    assert "draft" not in redacted
    assert "Project X overview" not in str(redacted)
