# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-5B — SQLite UserProfile durable reference qualification."""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from intergrax.memory.contracts.provider_durability_evidence import (
    MemoryProviderTrustedDurabilityStatus,
    durability_evidence_from_reopen_proof,
)
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    build_user_profile_admission_evidence_from_durable_qualification,
)
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_memory import (
    MemoryImportance,
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from tests.unit.memory.durable_provider_qualification_harness import (
    DurabilityQualificationMode,
    run_durable_user_profile_production_qualification,
    user_profile_qualification_request,
)
from tests.unit.memory.test_mem_ent13_provider_qualification import (
    _StaticFactory,
    _context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _sqlite_factory(
    db_path: Path,
) -> tuple[Callable[[], SQLiteUserProfileStore], Callable[[SQLiteUserProfileStore], Awaitable[None]]]:
    def _create() -> SQLiteUserProfileStore:
        return SQLiteUserProfileStore(str(db_path))

    async def _dispose(store: SQLiteUserProfileStore) -> None:
        store.close()

    return _create, _dispose


@pytest.mark.asyncio
async def test_sqlite_durable_harness_produces_production_admission_bundle(tmp_path: Path) -> None:
    db_path = tmp_path / "cert.db"
    create, dispose = _sqlite_factory(db_path)
    evidence = await run_durable_user_profile_production_qualification(
        descriptor=MemoryProviderDescriptor(
            provider_id="sqlite.user_profile",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context("mem-final-audit-5b-cert"),
        request=user_profile_qualification_request(),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_StaticFactory(create, dispose)),
        create_store=create,
        dispose_store=dispose,
        durability_mode=DurabilityQualificationMode.DURABLE_PERSISTENCE,
    )
    bundle = build_user_profile_admission_evidence_from_durable_qualification(
        canonical=evidence.canonical,
        reopen_passed=evidence.reopen_passed,
        delete_reopen_passed=evidence.delete_reopen_passed,
        production_durable_qualified=evidence.production_durable_qualified,
    )
    assert bundle.production_durable_qualified
    assert bundle.qualification_run_id == evidence.canonical.qualification_run_id


@pytest.mark.asyncio
async def test_delete_reopen_none_is_not_trusted_durable_evidence() -> None:
    mapped = durability_evidence_from_reopen_proof(
        provider_id="sqlite.user_profile",
        capability=MemoryProviderCapabilityKind.USER_PROFILE_STORE,
        qualification_run_id="run-x",
        reference_time_iso="2025-01-01T00:00:00+00:00",
        reopen_passed=True,
        delete_reopen_passed=None,
    )
    assert mapped.durability_status is MemoryProviderTrustedDurabilityStatus.UNKNOWN


@pytest.mark.asyncio
async def test_unicode_and_memory_entries_survive_cross_instance_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "unicode.db"
    marker = "Instrukcja: żółć 🚀\nmulti-line"
    profile = UserProfile(
        identity=UserIdentity(user_id="u-pl", display_name="Użytkownik"),
        preferences=UserPreferences(preferred_language="pl", tone="formal"),
        system_instructions=marker,
        memory_entries=[
            UserProfileMemoryEntry(
                entry_id="e1",
                content="Lubi kawę",
                kind=MemoryKind.PREFERENCE,
                importance=MemoryImportance.HIGH,
            ),
            UserProfileMemoryEntry(
                entry_id="e2",
                content="Emoji test 😀",
                kind=MemoryKind.USER_FACT,
                importance=MemoryImportance.MEDIUM,
            ),
        ],
    )
    store_a = SQLiteUserProfileStore(str(db_path))
    await store_a.save_profile(tenant_id="tenant-a", profile=profile)
    store_a.close()

    store_b = SQLiteUserProfileStore(str(db_path))
    loaded = await store_b.get_profile(tenant_id="tenant-a", user_id="u-pl")
    store_b.close()

    assert loaded.system_instructions == marker
    assert loaded.preferences.preferred_language == "pl"
    assert len(loaded.memory_entries) == 2
    assert loaded.memory_entries[0].content == "Lubi kawę"


@pytest.mark.asyncio
async def test_tenant_isolation_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "tenant.db"
    store_a = SQLiteUserProfileStore(str(db_path))
    await store_a.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="shared-user", display_name="A"),
            preferences=UserPreferences(),
        ),
    )
    await store_a.save_profile(
        tenant_id="tenant-b",
        profile=UserProfile(
            identity=UserIdentity(user_id="shared-user", display_name="B"),
            preferences=UserPreferences(),
        ),
    )
    store_a.close()

    store_b = SQLiteUserProfileStore(str(db_path))
    loaded_a = await store_b.get_profile(tenant_id="tenant-a", user_id="shared-user")
    loaded_b = await store_b.get_profile(tenant_id="tenant-b", user_id="shared-user")
    store_b.close()

    assert loaded_a.identity.display_name == "A"
    assert loaded_b.identity.display_name == "B"


@pytest.mark.asyncio
async def test_user_isolation_survives_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "user.db"
    store_a = SQLiteUserProfileStore(str(db_path))
    await store_a.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="user-a", display_name="User A"),
            preferences=UserPreferences(),
        ),
    )
    await store_a.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="user-b", display_name="User B"),
            preferences=UserPreferences(),
        ),
    )
    store_a.close()

    store_b = SQLiteUserProfileStore(str(db_path))
    assert (await store_b.get_profile(tenant_id="tenant-a", user_id="user-a")).identity.display_name == "User A"
    assert (await store_b.get_profile(tenant_id="tenant-a", user_id="user-b")).identity.display_name == "User B"
    store_b.close()


@pytest.mark.asyncio
async def test_update_version_survives_double_reopen(tmp_path: Path) -> None:
    db_path = tmp_path / "update.db"
    tenant, user = "tenant-a", "user-x"

    async def _save_version(version_marker: str) -> None:
        store = SQLiteUserProfileStore(str(db_path))
        await store.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id=user),
                preferences=UserPreferences(),
                system_instructions=version_marker,
            ),
        )
        store.close()

    await _save_version("v1")
    store_b = SQLiteUserProfileStore(str(db_path))
    assert (await store_b.get_profile(tenant_id=tenant, user_id=user)).system_instructions == "v1"
    store_b.close()

    await _save_version("v2")
    store_c = SQLiteUserProfileStore(str(db_path))
    assert (await store_c.get_profile(tenant_id=tenant, user_id=user)).system_instructions == "v2"
    store_c.close()


@pytest.mark.asyncio
async def test_delete_durability_does_not_remove_sibling_tenant(tmp_path: Path) -> None:
    db_path = tmp_path / "delete-sibling.db"
    store_a = SQLiteUserProfileStore(str(db_path))
    for tenant in ("tenant-a", "tenant-b"):
        await store_a.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id="u1", display_name=tenant),
                preferences=UserPreferences(),
            ),
        )
    await store_a.delete_profile(tenant_id="tenant-a", user_id="u1")
    store_a.close()

    store_b = SQLiteUserProfileStore(str(db_path))
    deleted = await store_b.get_profile(tenant_id="tenant-a", user_id="u1")
    sibling = await store_b.get_profile(tenant_id="tenant-b", user_id="u1")
    store_b.close()

    assert deleted.identity.display_name is None
    assert sibling.identity.display_name == "tenant-b"


@pytest.mark.asyncio
async def test_triple_instance_reopen_cycle(tmp_path: Path) -> None:
    db_path = tmp_path / "triple.db"
    for label in ("a", "b", "c"):
        store = SQLiteUserProfileStore(str(db_path))
        await store.save_profile(
            tenant_id="tenant-a",
            profile=UserProfile(
                identity=UserIdentity(user_id="u1"),
                preferences=UserPreferences(),
                system_instructions=f"cycle-{label}",
            ),
        )
        store.close()

    final = SQLiteUserProfileStore(str(db_path))
    loaded = await final.get_profile(tenant_id="tenant-a", user_id="u1")
    final.close()
    assert loaded.system_instructions == "cycle-c"


@pytest.mark.asyncio
async def test_close_is_idempotent_and_use_after_close_fails(tmp_path: Path) -> None:
    store = SQLiteUserProfileStore(str(tmp_path / "close.db"))
    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        await store.get_profile(tenant_id="t", user_id="u")


@pytest.mark.asyncio
async def test_failed_save_does_not_overwrite_committed_state(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("read-only chmod semantics are not reliable on Windows for this proof")
    db_path = tmp_path / "failed-save.db"
    store = SQLiteUserProfileStore(str(db_path))
    await store.save_profile(
        tenant_id="tenant-a",
        profile=UserProfile(
            identity=UserIdentity(user_id="u1"),
            preferences=UserPreferences(),
            system_instructions="committed-v1",
        ),
    )
    store.close()

    os.chmod(db_path, stat.S_IREAD)
    store_ro = SQLiteUserProfileStore(str(db_path))
    with pytest.raises(sqlite3.OperationalError):
        await store_ro.save_profile(
            tenant_id="tenant-a",
            profile=UserProfile(
                identity=UserIdentity(user_id="u1"),
                preferences=UserPreferences(),
                system_instructions="failed-v2",
            ),
        )
    store_ro.close()

    os.chmod(db_path, stat.S_IWRITE | stat.S_IREAD)
    store_reopen = SQLiteUserProfileStore(str(db_path))
    loaded = await store_reopen.get_profile(tenant_id="tenant-a", user_id="u1")
    store_reopen.close()
    assert loaded.system_instructions == "committed-v1"
