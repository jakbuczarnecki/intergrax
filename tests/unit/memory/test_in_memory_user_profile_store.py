# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.memory.stores.in_memory_user_profile_store import (
    InMemoryUserProfileStore,
)
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_profiles_are_isolated_between_tenants():
    store = InMemoryUserProfileStore()

    profile_a = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
    )

    profile_b = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
    )

    await store.save_profile(
        tenant_id="tenant-a",
        profile=profile_a,
    )

    await store.save_profile(
        tenant_id="tenant-b",
        profile=profile_b,
    )

    loaded_a = await store.get_profile(
        tenant_id="tenant-a",
        user_id="u1",
    )

    loaded_b = await store.get_profile(
        tenant_id="tenant-b",
        user_id="u1",
    )

    assert loaded_a.identity.user_id == "u1"
    assert loaded_b.identity.user_id == "u1"
    assert loaded_a is not loaded_b


@pytest.mark.asyncio
async def test_profile_not_visible_across_tenants():
    store = InMemoryUserProfileStore()

    profile = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
    )

    await store.save_profile(
        tenant_id="tenant-a",
        profile=profile,
    )

    loaded_a = await store.get_profile(
        tenant_id="tenant-a",
        user_id="u1",
    )

    loaded_b = await store.get_profile(
        tenant_id="tenant-b",
        user_id="u1",
    )

    # Tenants must not share profile instance
    assert loaded_a is not loaded_b
    assert loaded_a.identity.user_id == "u1"
    assert loaded_b.identity.user_id == "u1"


@pytest.mark.asyncio
async def test_delete_profile_is_tenant_scoped():
    store = InMemoryUserProfileStore()

    profile_a = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
    )

    profile_b = UserProfile(
        identity=UserIdentity(user_id="u1"),
        preferences=UserPreferences(),
    )

    await store.save_profile(
        tenant_id="tenant-a",
        profile=profile_a,
    )

    await store.save_profile(
        tenant_id="tenant-b",
        profile=profile_b,
    )

    await store.delete_profile(
        tenant_id="tenant-a",
        user_id="u1",
    )

    # After delete, tenant-a gets a new default profile
    reloaded_a = await store.get_profile(
        tenant_id="tenant-a",
        user_id="u1",
    )

    loaded_b = await store.get_profile(
        tenant_id="tenant-b",
        user_id="u1",
    )

    # tenant-a should not be the same object as original profile_a
    assert reloaded_a is not profile_a

    # tenant-b must still have its stored profile
    assert loaded_b is profile_b
