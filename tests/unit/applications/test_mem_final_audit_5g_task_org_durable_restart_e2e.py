# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5G — application composition Task/Organization durable restart E2E."""

from __future__ import annotations

from pathlib import Path

import pytest
from lab_application.host.settings import LabApplicationSettings

from intergrax.applications._shared.lab_environment_profile import build_lab_environment_profile
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications._shared.task_memory_wiring import wire_task_memory_from_profile
from intergrax.integrations.providers.relational_store.sqlite.bundle import create_sqlite_integration
from intergrax.runtime.organization.organization_profile import (
    OrganizationIdentity,
    OrganizationProfile,
)
from intergrax.runtime.organization.organization_profile_manager import OrganizationProfileManager
from intergrax.runtime.organization.stores.sqlite_organization_profile_store import (
    SQLiteOrganizationProfileStore,
)
from intergrax.runtime.task_memory.coordinator import TaskMemoryCoordinator
from intergrax.runtime.task_memory.stores.sqlite_task_memory_store import SQLiteTaskMemoryStore

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _lab_env(tmp_path: Path):
    settings = LabApplicationSettings(include_echo=True, include_mock_agents=False)
    env = build_lab_environment_profile(settings)
    env.integration_profile.options = {
        **(env.integration_profile.options or {}),
        "sqlite": {"data_dir": str(tmp_path)},
    }
    return env


@pytest.mark.asyncio
async def test_lab_task_memory_composition_survives_rebuild(tmp_path: Path) -> None:
    env = _lab_env(tmp_path)
    bundle = create_sqlite_integration(data_dir=tmp_path)
    wiring_a = wire_task_memory_from_profile(env, db_path=bundle.paths.task_memory)
    assert wiring_a.store is not None
    assert isinstance(wiring_a.store, SQLiteTaskMemoryStore)
    TaskMemoryCoordinator.write(
        wiring_a.store,
        tenant_id="lab",
        task_id="task-compose",
        namespace="qual",
        key="marker",
        value={"content": "compose-a"},
    )
    wiring_a.store.close()

    wiring_b = wire_task_memory_from_profile(env, db_path=bundle.paths.task_memory)
    assert wiring_b.store is not None
    loaded = TaskMemoryCoordinator.read(
        wiring_b.store,
        tenant_id="lab",
        task_id="task-compose",
        namespace="qual",
        key="marker",
    )
    wiring_b.store.close()
    assert loaded is not None
    assert loaded.value["content"] == "compose-a"


@pytest.mark.asyncio
async def test_lab_organization_memory_composition_survives_rebuild(tmp_path: Path) -> None:
    env = _lab_env(tmp_path)
    wiring_a = resolve_memory_platform_wiring(env)
    assert wiring_a.organization_profile_store is not None
    assert isinstance(wiring_a.organization_profile_store, SQLiteOrganizationProfileStore)
    manager_a = OrganizationProfileManager(wiring_a.organization_profile_store)
    await manager_a.save_profile(
        OrganizationProfile(
            identity=OrganizationIdentity(organization_id="lab-org", name="Lab Org"),
            system_instructions="org-compose-marker",
        ),
    )
    wiring_a.organization_profile_store.close()

    wiring_b = resolve_memory_platform_wiring(env)
    assert wiring_b.organization_profile_store is not None
    manager_b = OrganizationProfileManager(wiring_b.organization_profile_store)
    loaded = await manager_b.get_profile("lab-org")
    wiring_b.organization_profile_store.close()
    assert loaded.system_instructions == "org-compose-marker"
