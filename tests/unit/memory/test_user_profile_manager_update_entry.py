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
    UserProfileMemoryEntryNotFoundError,
)

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_update_memory_entry_not_found_raises_and_skips_side_effects() -> None:
    store = MagicMock()
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
        memory_entries=[],
    )
    store.get_profile = AsyncMock(return_value=profile)
    store.save_profile = AsyncMock()
    mgr = UserProfileManager(store)

    with pytest.raises(UserProfileMemoryEntryNotFoundError):
        await mgr.update_memory_entry("u1", "missing-entry", content="x")

    store.save_profile.assert_not_awaited()
