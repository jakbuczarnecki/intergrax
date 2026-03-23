# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from pathlib import Path

from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from testing_support.builder import prepare_sqlite_db


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_sqlite_organization_profile_store_persistence() -> None:
    db_path: Path = prepare_sqlite_db("organization_test.db")

    store = SQLiteOrganizationProfileStore(db_path=str(db_path))

    # ------------------------------------------------------------------
    # 1. Default profile returned when not exists
    # ------------------------------------------------------------------

    profile = await store.get_profile("intergrax")

    assert profile.identity.organization_id == "intergrax"
    assert profile.identity.name == "intergrax"
    assert profile.system_instructions is None

    # ------------------------------------------------------------------
    # 2. Modify and save profile
    # ------------------------------------------------------------------

    profile.identity.name = "intergrax"
    profile.preferences.tone_of_voice = "confident-professional"
    profile.preferences.allow_tools = False
    profile.system_instructions = "Custom org-level instructions."
    profile.domain_summary = "AI agent factory platform."
    profile.tags = ["ai", "agents"]

    await store.save_profile(profile)

    # ------------------------------------------------------------------
    # 3. Reload and verify persistence
    # ------------------------------------------------------------------

    reloaded = await store.get_profile("intergrax")

    assert reloaded.identity.name == "intergrax"
    assert reloaded.preferences.tone_of_voice == "confident-professional"
    assert reloaded.preferences.allow_tools is False
    assert reloaded.system_instructions == "Custom org-level instructions."
    assert reloaded.domain_summary == "AI agent factory platform."
    assert reloaded.tags == ["ai", "agents"]

    # ------------------------------------------------------------------
    # 4. Delete profile
    # ------------------------------------------------------------------

    await store.delete_profile("intergrax")

    after_delete = await store.get_profile("intergrax")

    # Contract: must return initialized default profile
    assert after_delete.identity.organization_id == "intergrax"
    assert after_delete.identity.name == "intergrax"
    assert after_delete.system_instructions is None