# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_consolidation import (
    SessionConsolidationReason,
    SessionMemoryConsolidationCoordinator,
)
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationService,
)

pytestmark = pytest.mark.gate


def test_mid_session_requires_interval_and_cooldown() -> None:
    session = ChatSession(id="s1", tenant_id="t1", user_id="u1")
    coord = SessionMemoryConsolidationCoordinator(
        service=MagicMock(spec=SessionMemoryConsolidationService),
        user_turns_interval=4,
        cooldown_seconds=60,
    )
    assert not coord.should_consolidate_mid_session(session, user_turns=3)
    assert coord.should_consolidate_mid_session(session, user_turns=4)


def test_mid_session_blocked_by_cooldown() -> None:
    session = ChatSession(id="s1", tenant_id="t1", user_id="u1")
    session.last_consolidated_at = datetime.now(timezone.utc)
    coord = SessionMemoryConsolidationCoordinator(
        service=MagicMock(spec=SessionMemoryConsolidationService),
        user_turns_interval=4,
        cooldown_seconds=3600,
    )
    assert not coord.should_consolidate_mid_session(session, user_turns=4)


@pytest.mark.asyncio
async def test_consolidate_delegates_to_service() -> None:
    service = MagicMock(spec=SessionMemoryConsolidationService)
    service.consolidate_session = AsyncMock(return_value=[])
    coord = SessionMemoryConsolidationCoordinator(
        service=service,
        user_turns_interval=4,
        cooldown_seconds=0,
    )
    diag = await coord.consolidate(
        user_id="u1",
        session_id="s1",
        messages=[],
    )
    assert diag is not None
    service.consolidate_session.assert_awaited_once()


def test_apply_consolidation_metadata() -> None:
    session = ChatSession(id="s1", tenant_id="t1", user_id="u1")
    coord = SessionMemoryConsolidationCoordinator(
        service=None,
        user_turns_interval=0,
        cooldown_seconds=0,
    )
    coord.apply_consolidation_metadata(
        session,
        reason=SessionConsolidationReason.MID_SESSION,
        turn=8,
    )
    assert session.last_consolidated_reason == "mid_session"
    assert session.last_consolidated_turn == 8
    assert session.needs_user_instructions_refresh is True
