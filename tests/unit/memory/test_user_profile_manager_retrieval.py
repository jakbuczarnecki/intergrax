# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk, RetrievalResult, RetrievalTrace

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_search_longterm_uses_retrieval_service() -> None:
    store = MagicMock()
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
        memory_entries=[
            UserProfileMemoryEntry(entry_id="e1", content="fact", kind="user_fact"),
        ],
    )
    store.get_profile = AsyncMock(return_value=profile)

    service = MagicMock()
    service.retrieve.return_value = RetrievalResult(
        chunks=[
            RetrievalChunk(
                id="e1",
                text="fact",
                score=0.9,
                metadata={"entry_id": "e1", "user_id": "u1"},
            )
        ],
        used=True,
        reason="hits",
        trace=RetrievalTrace(),
    )

    mgr = UserProfileManager(store, retrieval_service=service)
    out = await mgr.search_longterm_memory("u1", "query")
    assert out["used_longterm"] is True
    assert len(out["hits"]) == 1
    service.retrieve.assert_called_once()
