# © Artur Czarnecki. All rights reserved.

"""Unit tests: Legal API v1 ↔ RuntimeRequest/Answer mapping."""

from __future__ import annotations

from legal_application.serving.runtime_bridge import LegalApiV1RuntimeMapper
from legal_application.serving.schemas import LegalChatRequestV1
from intergrax.fastapi_core.context import RequestContext
from intergrax.runtime.task.task_run_bridge import mint_intake_execution_identity
from intergrax.runtime.nexus.responses.response_schema import (
    RouteInfo,
    RuntimeAnswer,
    RuntimeStats,
    StopReason,
)

import pytest

pytestmark = pytest.mark.unit


def test_legal_chat_request_maps_to_runtime_request() -> None:
    ctx = RequestContext(
        request_id="req-1",
        path="/v1/legal/chat",
        method="POST",
        tenant_id=None,
        user_id=None,
        auth=None,
    )
    body = LegalChatRequestV1(
        message="Hello",
        session_id="sess-1",
        workspace_id="ws-1",
        tenant_id="ten-1",
        user_id="usr-1",
        metadata={"k": "v"},
    )
    mapper = LegalApiV1RuntimeMapper()
    task_id, run_id = mint_intake_execution_identity()
    rt = mapper.to_runtime_request(
        body,
        http_context=ctx,
        default_agent_id="legal-default",
        tenant_id="ten-1",
        user_id="usr-1",
        task_id=task_id,
        run_id=run_id,
    )
    assert rt.agent_id == "legal-default"
    assert rt.message == "Hello"
    assert rt.session_id == "sess-1"
    assert rt.workspace_id == "ws-1"
    assert rt.tenant_id == "ten-1"
    assert rt.user_id == "usr-1"
    assert rt.task_id == task_id
    assert rt.run_id == run_id
    assert rt.metadata.get("api") == {"product": "legal_agent", "version": "1"}
    assert rt.metadata.get("http_request_id") == "req-1"


def test_runtime_answer_maps_to_legal_response_without_trace() -> None:
    ctx = RequestContext(
        request_id="req-2",
        path="/x",
        method="POST",
        tenant_id="t",
        user_id="u",
        auth=None,
    )
    answer = RuntimeAnswer(
        answer="Done",
        stop_reason=StopReason.COMPLETED,
        run_id="run-xyz",
        route=RouteInfo(strategy="legal_agent", extra={"a": 1}),
        stats=RuntimeStats(total_tokens=10, input_tokens=4, output_tokens=6),
        trace_events=[],
    )
    out = LegalApiV1RuntimeMapper().to_legal_chat_response(
        answer, http_context=ctx, include_trace=False
    )
    assert out.request_id == "req-2"
    assert out.run_id == "run-xyz"
    assert out.answer == "Done"
    assert out.stop_reason == "completed"
    assert out.route["strategy"] == "legal_agent"
    assert out.stats["total_tokens"] == 10
    assert out.trace_events is None
