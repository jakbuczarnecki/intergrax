# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5G — Organization Memory SQLite durable restart qualification."""

from __future__ import annotations

import pytest

from intergrax.runtime.persistence.sqlite_opens import (
    open_organization_profile_store_at,
)
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_profile_instructions import (
    SessionProfileInstructionResolver,
)
from intergrax.runtime.organization.organization_profile import (
    OrganizationIdentity,
    OrganizationProfile,
)
from intergrax.runtime.organization.organization_profile_manager import (
    OrganizationProfileManager,
)
from tests.integration.memory.e2e.mem_final_audit_5g_sqlite_restart_support import (
    parse_worker_json,
    run_restart_worker,
    write_fixture,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate, pytest.mark.asyncio]

_MARKER_V1 = "Org: żółć 🏢 v1"
_MARKER_V2 = "Org: żółć 🏢 v2"


async def _save(
    db_path,
    *,
    organization_id: str,
    system_instructions: str,
    tags: list[str] | None = None,
) -> None:
    store = open_organization_profile_store_at(db_path)
    manager = OrganizationProfileManager(store)
    try:
        await manager.save_profile(
            OrganizationProfile(
                identity=OrganizationIdentity(
                    organization_id=organization_id,
                    name=organization_id,
                ),
                system_instructions=system_instructions,
                tags=tags or ["qual"],
            ),
        )
    finally:
        store.close()


async def _load(db_path, organization_id: str) -> OrganizationProfile:
    store = open_organization_profile_store_at(db_path)
    manager = OrganizationProfileManager(store)
    try:
        return await manager.get_profile(organization_id)
    finally:
        store.close()


@pytest.mark.unit
async def test_organization_write_reopen_read(tmp_path) -> None:
    db_path = tmp_path / "org.db"
    await _save(db_path, organization_id="org-a", system_instructions=_MARKER_V1)
    loaded = await _load(db_path, "org-a")
    assert loaded.system_instructions == _MARKER_V1
    assert loaded.tags == ["qual"]


@pytest.mark.unit
async def test_organization_update_survives_double_reopen(tmp_path) -> None:
    db_path = tmp_path / "org-update.db"
    await _save(db_path, organization_id="org-a", system_instructions=_MARKER_V1)
    assert (await _load(db_path, "org-a")).system_instructions == _MARKER_V1
    await _save(db_path, organization_id="org-a", system_instructions=_MARKER_V2)
    loaded = await _load(db_path, "org-a")
    assert loaded.system_instructions == _MARKER_V2
    assert loaded.system_instructions != _MARKER_V1


@pytest.mark.unit
async def test_organization_delete_resets_to_default_after_reopen(tmp_path) -> None:
    db_path = tmp_path / "org-delete.db"
    await _save(db_path, organization_id="org-a", system_instructions=_MARKER_V1)
    store = open_organization_profile_store_at(db_path)
    manager = OrganizationProfileManager(store)
    try:
        await manager.delete_profile("org-a")
    finally:
        store.close()
    loaded = await _load(db_path, "org-a")
    assert loaded.system_instructions is None
    assert loaded.identity.organization_id == "org-a"


@pytest.mark.unit
async def test_organization_triple_reopen_cycle(tmp_path) -> None:
    db_path = tmp_path / "org-triple.db"
    for idx in range(3):
        await _save(
            db_path,
            organization_id="org-a",
            system_instructions=f"cycle-{idx}",
        )
    loaded = await _load(db_path, "org-a")
    assert loaded.system_instructions == "cycle-2"


@pytest.mark.unit
async def test_organization_isolation_after_restart(tmp_path) -> None:
    db_path = tmp_path / "org-iso.db"
    await _save(db_path, organization_id="org-a", system_instructions="A-only")
    await _save(db_path, organization_id="org-b", system_instructions="B-only")
    a = await _load(db_path, "org-a")
    b = await _load(db_path, "org-b")
    assert a.system_instructions == "A-only"
    assert b.system_instructions == "B-only"


@pytest.mark.unit
async def test_organization_close_is_idempotent_and_use_after_close_fails(
    tmp_path,
) -> None:
    store = open_organization_profile_store_at(tmp_path / "close.db")
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        await store.get_profile("org-a")


@pytest.mark.unit
async def test_organization_schema_idempotent_on_reopen(tmp_path) -> None:
    db_path = tmp_path / "org-schema.db"
    await _save(db_path, organization_id="org-a", system_instructions=_MARKER_V1)
    open_organization_profile_store_at(db_path).close()
    open_organization_profile_store_at(db_path).close()
    assert (await _load(db_path, "org-a")).system_instructions == _MARKER_V1


@pytest.mark.unit
async def test_organization_wrong_db_path_does_not_read_peer_data(tmp_path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    await _save(db_a, organization_id="org-a", system_instructions="secret")
    loaded = await _load(db_b, "org-a")
    assert loaded.system_instructions is None


@pytest.mark.unit
async def test_session_tenant_id_maps_to_organization_authority(tmp_path) -> None:
    db_path = tmp_path / "tenant-map.db"
    tenant_id = "tenant-as-org"
    await _save(db_path, organization_id=tenant_id, system_instructions="tenant-bound")
    store = open_organization_profile_store_at(db_path)
    manager = OrganizationProfileManager(store)
    resolver = SessionProfileInstructionResolver(organization_profile_manager=manager)
    session = ChatSession(
        id="sess-5g",
        tenant_id=tenant_id,
        user_id="user-1",
    )
    try:
        instructions = await resolver.org_instructions_for_session(session)
    finally:
        store.close()
    assert instructions == "tenant-bound"


@pytest.mark.unit
async def test_organization_fresh_process_save_read(tmp_path) -> None:
    db_path = tmp_path / "org-process.db"
    fixture = write_fixture(
        tmp_path,
        "org-fixture.json",
        {
            "organization_id": "org-proc",
            "system_instructions": _MARKER_V1,
            "tags": ["proc"],
        },
    )
    write_proc = run_restart_worker("org_save", db_path=db_path, fixture_path=fixture)
    assert write_proc.returncode == 0, write_proc.stderr

    read_proc = run_restart_worker("org_read", db_path=db_path, fixture_path=fixture)
    assert read_proc.returncode == 0, read_proc.stderr
    payload = parse_worker_json(read_proc.stdout)
    assert payload["system_instructions"] == _MARKER_V1
    assert payload["tags"] == ["proc"]
