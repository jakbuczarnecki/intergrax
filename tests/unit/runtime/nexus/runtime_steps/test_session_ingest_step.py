# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.session_and_ingest_step import SessionAndIngestStep
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from tests._support.builder import build_runtime_state_for_tests


@dataclass
class _FakeSession:
    id: str
    user_id: str
    tenant_id: str
    workspace_id: str


@dataclass
class _FakeAppendResult:
    consolidation_diag: object | None = None


@dataclass
class _FakeIngestionResult:
    attachment_id: str
    attachment_type: str
    num_chunks: int
    vector_ids: List[str]


class _FakeSessionManager(SessionStorage):
    def __init__(self, existing_session=None):
        self.existing_session = existing_session
        self.created_session = None
        self.append_called = False

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> Optional[ChatSession]:
        return self.existing_session

    async def create_session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChatSession:
        self.created_session = _FakeSession(
            id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        return self.created_session

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        self.append_called = True
        return _FakeAppendResult()

class _FakeIngestionService:
    def __init__(self, results):
        self.results = results
        self.called = False

    async def ingest_attachments_for_session(self, **kwargs):
        self.called = True
        return self.results


@pytest.mark.asyncio
async def test_session_loaded_when_exists():
    state = build_runtime_state_for_tests(run_id="run-1")

    existing = _FakeSession("s1", "u1", "t1", "w1")
    state.context.session_manager = _FakeSessionManager(existing_session=existing)

    await SessionAndIngestStep().run(state)

    assert state.session.id == "s1"
    assert state.context.session_manager.append_called is True


@pytest.mark.asyncio
async def test_session_created_when_missing():
    state = build_runtime_state_for_tests(run_id="run-1")

    state.context.session_manager = _FakeSessionManager(existing_session=None)

    await SessionAndIngestStep().run(state)

    assert state.context.session_manager.created_session is not None
    assert state.session.id == state.request.session_id


@pytest.mark.asyncio
async def test_ingestion_happens_when_attachments_present():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.attachments = ["file1"]

    state.context.session_manager = _FakeSessionManager(
        existing_session=_FakeSession("s1", "u1", "t1", "w1")
    )

    ingest_results = [
        _FakeIngestionResult("a1", "pdf", 5, ["v1", "v2"])
    ]
    state.context.ingestion_service = _FakeIngestionService(ingest_results)

    await SessionAndIngestStep().run(state)

    assert state.context.ingestion_service.called is True
    assert state.ingestion_results == ingest_results


@pytest.mark.asyncio
async def test_ingestion_requires_service():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.request.attachments = ["file1"]

    state.context.session_manager = _FakeSessionManager(
        existing_session=_FakeSession("s1", "u1", "t1", "w1")
    )
    state.context.ingestion_service = None

    with pytest.raises(ValueError):
        await SessionAndIngestStep().run(state)


@pytest.mark.asyncio
async def test_user_message_appended_to_session():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.context.session_manager = _FakeSessionManager(
        existing_session=_FakeSession("s1", "u1", "t1", "w1")
    )

    await SessionAndIngestStep().run(state)
    
    assert state.session.id == "s1"
