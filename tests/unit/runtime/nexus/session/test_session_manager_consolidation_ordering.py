# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_manager import SessionManager

pytestmark = pytest.mark.gate


@dataclass
class _RecordingConsolidationService:
    calls: list[tuple[str, str, tuple[str, ...]]] = field(default_factory=list)

    async def consolidate_session(
        self,
        user_id: str,
        session_id: str,
        messages: Sequence[ChatMessage],
        *,
        run_id: str | None = None,
    ) -> object:
        contents = tuple((message.content or "") for message in messages)
        self.calls.append((user_id, session_id, contents))
        return []


class _FailingAppendStorage(InMemorySessionStorage):
    async def append_message(self, *, tenant_id: str, session_id: str, message: ChatMessage):
        raise RuntimeError("append failed")


@pytest.mark.asyncio
async def test_mid_session_consolidation_history_includes_triggering_message() -> None:
    storage = InMemorySessionStorage()
    recorder = _RecordingConsolidationService()
    manager = SessionManager(
        storage,
        session_memory_consolidation_service=recorder,  # type: ignore[arg-type]
        user_turns_consolidation_interval=2,
        consolidation_cooldown_seconds=0,
        memory_consolidation_mode="auto",
    )
    tenant_id = "tenant-1"
    session = await manager.create_session(
        tenant_id=tenant_id,
        user_id="user-1",
        session_id="sess-1",
    )
    await manager.append_message(
        tenant_id=tenant_id,
        session_id=session.id,
        message=ChatMessage(role="user", content="message 1"),
    )
    await manager.append_message(
        tenant_id=tenant_id,
        session_id=session.id,
        message=ChatMessage(role="user", content="message 2"),
    )

    assert len(recorder.calls) == 1
    _, session_id, contents = recorder.calls[0]
    assert session_id == "sess-1"
    assert contents == ("message 1", "message 2")


@pytest.mark.asyncio
async def test_append_failure_does_not_invoke_consolidation() -> None:
    storage = _FailingAppendStorage()
    recorder = _RecordingConsolidationService()
    manager = SessionManager(
        storage,
        session_memory_consolidation_service=recorder,  # type: ignore[arg-type]
        user_turns_consolidation_interval=2,
        consolidation_cooldown_seconds=0,
        memory_consolidation_mode="auto",
    )
    tenant_id = "tenant-1"
    session = ChatSession(
        id="sess-fail",
        user_id="user-1",
        tenant_id=tenant_id,
    )
    await storage.save_session(session)

    with pytest.raises(RuntimeError, match="append failed"):
        await manager.append_message(
            tenant_id=tenant_id,
            session_id=session.id,
            message=ChatMessage(role="user", content="message 1"),
        )
    assert recorder.calls == []


@pytest.mark.asyncio
async def test_close_session_invokes_consolidation_with_tenant_scoped_history() -> None:
    storage = InMemorySessionStorage()
    recorder = _RecordingConsolidationService()
    manager = SessionManager(
        storage,
        session_memory_consolidation_service=recorder,  # type: ignore[arg-type]
        memory_consolidation_mode="auto",
    )
    tenant_id = "tenant-close"
    session = await manager.create_session(
        tenant_id=tenant_id,
        user_id="user-close",
        session_id="sess-close",
    )
    await manager.append_message(
        tenant_id=tenant_id,
        session_id=session.id,
        message=ChatMessage(role="user", content="close-turn"),
    )
    await manager.close_session(tenant_id=tenant_id, session_id=session.id)

    assert len(recorder.calls) == 1
    user_id, session_id, contents = recorder.calls[0]
    assert user_id == "user-close"
    assert session_id == "sess-close"
    assert contents == ("close-turn",)
