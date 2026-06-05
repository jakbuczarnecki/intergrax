# © Artur Czarnecki. All rights reserved.

"""MEM-2.1: SQLiteUserProfileStore CRUD round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.gate
async def test_sqlite_user_profile_store_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = SQLiteUserProfileStore(db_path=str(tmp_path / "user_profiles.db"))
    profile = UserProfile(
        identity=UserIdentity(user_id="u1", display_name="Ada"),
        preferences=UserPreferences(preferred_language="pl", tone="concise"),
        memory_entries=[
            UserProfileMemoryEntry(
                entry_id="mem-1",
                content="Prefers technical answers",
                kind=MemoryKind.PREFERENCE,
                importance=MemoryImportance.HIGH,
                title="Tone",
            )
        ],
        system_instructions="Be concise.",
    )

    await store.save_profile(tenant_id="tenant-a", profile=profile)
    loaded = await store.get_profile(tenant_id="tenant-a", user_id="u1")

    assert loaded.identity.user_id == "u1"
    assert loaded.identity.display_name == "Ada"
    assert loaded.preferences.preferred_language == "pl"
    assert loaded.system_instructions == "Be concise."
    assert len(loaded.memory_entries) == 1
    assert loaded.memory_entries[0].content == "Prefers technical answers"
    assert loaded.memory_entries[0].kind == MemoryKind.PREFERENCE


@pytest.mark.asyncio
@pytest.mark.gate
async def test_sqlite_user_profile_store_isolates_tenants(tmp_path: Path) -> None:
    store = SQLiteUserProfileStore(db_path=str(tmp_path / "user_profiles.db"))
    profile_a = UserProfile(
        identity=UserIdentity(user_id="u1", display_name="Tenant A"),
        preferences=UserPreferences(),
    )
    profile_b = UserProfile(
        identity=UserIdentity(user_id="u1", display_name="Tenant B"),
        preferences=UserPreferences(),
    )

    await store.save_profile(tenant_id="tenant-a", profile=profile_a)
    await store.save_profile(tenant_id="tenant-b", profile=profile_b)

    loaded_a = await store.get_profile(tenant_id="tenant-a", user_id="u1")
    loaded_b = await store.get_profile(tenant_id="tenant-b", user_id="u1")

    assert loaded_a.identity.display_name == "Tenant A"
    assert loaded_b.identity.display_name == "Tenant B"


@pytest.mark.asyncio
@pytest.mark.gate
async def test_sqlite_user_profile_store_delete_is_tenant_scoped(tmp_path: Path) -> None:
    store = SQLiteUserProfileStore(db_path=str(tmp_path / "user_profiles.db"))
    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
    )

    await store.save_profile(tenant_id="tenant-a", profile=profile)
    await store.save_profile(tenant_id="tenant-b", profile=profile)
    await store.delete_profile(tenant_id="tenant-a", user_id="u1")

    reloaded_a = await store.get_profile(tenant_id="tenant-a", user_id="u1")
    loaded_b = await store.get_profile(tenant_id="tenant-b", user_id="u1")

    assert reloaded_a.identity.user_id == "u1"
    assert reloaded_a.identity.display_name is None
    assert loaded_b.identity.user_id == "u1"


@pytest.mark.asyncio
async def test_sqlite_user_profile_store_returns_default_for_unknown_user(tmp_path: Path) -> None:
    store = SQLiteUserProfileStore(db_path=str(tmp_path / "user_profiles.db"))

    profile = await store.get_profile(tenant_id="tenant-a", user_id="new-user")

    assert profile.identity.user_id == "new-user"
    assert profile.memory_entries == []
