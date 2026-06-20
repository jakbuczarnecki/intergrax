# © Artur Czarnecki. All rights reserved.

"""MEM-MAINT-01..04 depth tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.memory.cognitive_store_mapping import cognitive_store_for_kind, org_scope_for_kind
from intergrax.memory.memory_temporal import filter_active_memory_entries, is_memory_entry_active
from intergrax.memory.org_memory_maturity import evaluate_org_memory_maturity, org_memory_entry
from intergrax.memory.org_memory_scope import OrgMemoryScope
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)

pytestmark = pytest.mark.gate


def test_procedural_kind_maps_to_procedural_ltm_store() -> None:
    assert cognitive_store_for_kind(MemoryKind.PROCEDURAL) == "procedural_ltm"
    assert org_scope_for_kind(MemoryKind.PROCEDURAL) is OrgMemoryScope.ORG_PROCEDURAL


@pytest.mark.asyncio
async def test_procedural_memory_write_and_read_path() -> None:
    store = MagicMock()
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
        memory_entries=[],
    )
    store.get_profile = AsyncMock(return_value=profile)
    store.save_profile = AsyncMock()

    mgr = UserProfileManager(store)
    entry = await mgr.add_memory_entry(
        "u1",
        UserProfileMemoryEntry(
            content="Always run pytest -m gate before merge",
            kind=MemoryKind.PROCEDURAL,
            metadata={"cognitive_store": cognitive_store_for_kind(MemoryKind.PROCEDURAL)},
        ),
    )
    assert entry.kind is MemoryKind.PROCEDURAL
    assert profile.memory_entries[-1].content.startswith("Always run pytest")
    store.save_profile.assert_awaited()


def test_org_memory_maturity_checklist_passes_with_scope_metadata() -> None:
    profile = UserProfile(
        identity=UserIdentity(user_id="org-1"),
        preferences=UserPreferences(),
        memory_entries=[
            org_memory_entry(
                content="Team uses uv for dependency management",
                kind=MemoryKind.ORG_FACT,
                scope=OrgMemoryScope.ORG_KNOWLEDGE,
            ),
        ],
    )
    result = evaluate_org_memory_maturity(profile)
    assert result.passed is True


def test_org_memory_maturity_flags_missing_scope() -> None:
    profile = UserProfile(
        identity=UserIdentity(user_id="org-1"),
        preferences=UserPreferences(),
        memory_entries=[
            UserProfileMemoryEntry(content="undocumented org fact", kind=MemoryKind.ORG_FACT),
        ],
    )
    result = evaluate_org_memory_maturity(profile)
    assert result.passed is False
    assert result.violations


def test_temporal_validity_filters_expired_ltm_facts() -> None:
    now = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    expired = UserProfileMemoryEntry(
        content="old policy",
        valid_until=(now - timedelta(days=1)).isoformat(),
    )
    active = UserProfileMemoryEntry(
        content="current policy",
        valid_from=(now - timedelta(days=1)).isoformat(),
        valid_until=(now + timedelta(days=30)).isoformat(),
    )
    assert is_memory_entry_active(expired, as_of=now) is False
    assert is_memory_entry_active(active, as_of=now) is True
    filtered = filter_active_memory_entries([expired, active], as_of=now)
    assert filtered == [active]
