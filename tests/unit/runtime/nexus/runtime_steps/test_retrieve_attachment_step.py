# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.retrieve_attachments_step import RetrieveAttachmentsStep
from tests._support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class _FakeIngestionService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.called = False

    async def search_session_attachments(self, **kwargs):
        self.called = True
        if self.error:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_attachments_step_skips_when_service_missing():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.ingestion_service = None
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1", "workspace_id": "w1"})()

    before_msgs = list(state.messages_for_llm)

    await RetrieveAttachmentsStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_attachments_context is False
    assert state.attachments_chunks_count == 0


@pytest.mark.asyncio
async def test_attachments_step_skips_when_no_session():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.ingestion_service = _FakeIngestionService(result=None)
    state.session = None

    before_msgs = list(state.messages_for_llm)

    await RetrieveAttachmentsStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_attachments_context is False


@pytest.mark.asyncio
async def test_attachments_step_no_hits_not_used():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1", "workspace_id": "w1"})()
    state.context.ingestion_service = _FakeIngestionService(
        result={"used": False, "hits": []}
    )

    before_msgs = list(state.messages_for_llm)

    await RetrieveAttachmentsStep().run(state)

    assert state.messages_for_llm == before_msgs
    assert state.used_attachments_context is False
    assert state.attachments_chunks_count == 0


@pytest.mark.asyncio
async def test_attachments_step_hits_used_injects_context():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="u1"),
    ]

    state.session = type("S", (), {"id": "s1", "tenant_id": "t1", "workspace_id": "w1"})()
    chunks = [{"text": "chunk1"}, {"text": "chunk2"}]
    state.context.ingestion_service = _FakeIngestionService(
        result={"used": True, "hits": chunks}
    )

    await RetrieveAttachmentsStep().run(state)

    assert state.used_attachments_context is True
    assert state.attachments_chunks_count == 2

    assert "SESSION ATTACHMENTS (retrieved):" in state.messages_for_llm[-2].content
    assert state.messages_for_llm[-1].content == "u1"

    assert state.tools_context_parts[-1].startswith("SESSION ATTACHMENTS:\n")


@pytest.mark.asyncio
async def test_attachments_step_format_returns_empty_treated_as_unused():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1", "workspace_id": "w1"})()
    state.context.ingestion_service = _FakeIngestionService(
        result={"used": True, "hits": []}
    )

    await RetrieveAttachmentsStep().run(state)

    assert state.used_attachments_context is False
    assert state.attachments_chunks_count == 0


@pytest.mark.asyncio
async def test_attachments_step_propagates_error():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1", "workspace_id": "w1"})()
    state.context.ingestion_service = _FakeIngestionService(
        error=RuntimeError("fail")
    )

    with pytest.raises(RuntimeError):
        await RetrieveAttachmentsStep().run(state)
