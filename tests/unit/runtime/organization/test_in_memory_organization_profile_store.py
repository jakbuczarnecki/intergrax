# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.organization.stores.in_memory_organization_profile_store import (
    InMemoryOrganizationProfileStore,
)

pytestmark = pytest.mark.unit
    


@pytest.mark.asyncio
async def test_in_memory_organization_profile_store_basic_flow() -> None:
    store = InMemoryOrganizationProfileStore()

    # ------------------------------------------------------------------
    # 1. Default profile
    # ------------------------------------------------------------------

    profile = await store.get_profile("intergrax")

    assert profile.identity.organization_id == "intergrax"
    assert profile.identity.name == "intergrax"
    assert profile.system_instructions is None

    # Default profile should now be stored internally
    assert "intergrax" in store.list_organization_ids()

    # ------------------------------------------------------------------
    # 2. Modify and save
    # ------------------------------------------------------------------

    profile.identity.name = "intergrax"
    profile.preferences.allow_tools = False
    profile.system_instructions = "In-memory instructions"
    profile.tags = ["ai", "agents"]

    await store.save_profile(profile)

    reloaded = await store.get_profile("intergrax")

    assert reloaded.preferences.allow_tools is False
    assert reloaded.system_instructions == "In-memory instructions"
    assert reloaded.tags == ["ai", "agents"]

    # ------------------------------------------------------------------
    # 3. Delete
    # ------------------------------------------------------------------

    await store.delete_profile("intergrax")

    assert "intergrax" not in store.list_organization_ids()

    # Contract: get_profile must recreate default
    new_profile = await store.get_profile("intergrax")

    assert new_profile.identity.organization_id == "intergrax"
    assert new_profile.identity.name == "intergrax"
    assert new_profile.system_instructions is None
