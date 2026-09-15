# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryLifecycleOperation,
    MemoryReconciliationDisposition,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.memory.user_profile_memory_lifecycle import UserProfileMemoryLifecyclePartialError

pytestmark = pytest.mark.gate

FAKE_PROJECTION_ID = "test_memory_projection"


@dataclass
class RecordingMemoryProjection:
    projection_id: str = FAKE_PROJECTION_ID
    upsert_calls: list[tuple[str, str]] = field(default_factory=list)
    delete_calls: list[tuple[str, ...]] = field(default_factory=list)
    reconcile_calls: int = 0
    fail_upsert: bool = False
    fail_delete: bool = False
    indexed_entry_ids: set[str] = field(default_factory=set)

    async def upsert_memory_entry(
        self,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> None:
        if self.fail_upsert:
            raise TimeoutError("projection unavailable")
        self.upsert_calls.append((user_id, entry.entry_id))
        self.indexed_entry_ids.add(entry.entry_id)

    async def delete_memory_entries(self, entry_ids: Sequence[str]) -> None:
        if self.fail_delete:
            raise TimeoutError("projection unavailable")
        self.delete_calls.append(tuple(entry_ids))
        for entry_id in entry_ids:
            self.indexed_entry_ids.discard(entry_id)

    async def reconcile(self, context: UserProfileMemoryReconciliationContext) -> None:
        self.reconcile_calls += 1
        expected = set(context.authoritative_active_entry_ids)
        orphans = self.indexed_entry_ids - expected
        if orphans:
            await self.delete_memory_entries(tuple(sorted(orphans)))
        if context.profile is None:
            if self.indexed_entry_ids:
                await self.delete_memory_entries(tuple(sorted(self.indexed_entry_ids)))
            return
        for entry in context.profile.memory_entries:
            if entry.deleted:
                continue
            if entry.entry_id not in self.indexed_entry_ids:
                await self.upsert_memory_entry(context.user_id, entry)


def _manager(
    store: InMemoryUserProfileStore,
    projection: RecordingMemoryProjection,
) -> UserProfileManager:
    return UserProfileManager(
        store,
        tenant_id="tenant-a",
        memory_projections=(projection,),
    )


async def _seed_profile(store: InMemoryUserProfileStore, user_id: str, entry_id: str) -> None:
    profile = UserProfile(
        identity=UserIdentity(user_id=user_id),
        preferences=UserPreferences(),
        memory_entries=[
            UserProfileMemoryEntry(entry_id=entry_id, content="fact one"),
        ],
    )
    await store.save_profile(tenant_id="tenant-a", profile=profile)


@pytest.mark.asyncio
async def test_write_success_primary_and_projection() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection()
    mgr = _manager(store, projection)

    entry = await mgr.add_memory_entry("u1", "remember this")

    assert entry.content == "remember this"
    assert projection.upsert_calls == [("u1", entry.entry_id)]


@pytest.mark.asyncio
async def test_write_primary_failure_skips_projection() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection()
    mgr = _manager(store, projection)

    async def fail_save(**kwargs: object) -> None:
        raise OSError("primary down")

    store.save_profile = fail_save  # type: ignore[method-assign]

    with pytest.raises(OSError):
        await mgr.add_memory_entry("u1", "x")

    assert projection.upsert_calls == []


@pytest.mark.asyncio
async def test_write_projection_failure_leaves_primary_and_raises_partial() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection(fail_upsert=True)
    mgr = _manager(store, projection)

    with pytest.raises(UserProfileMemoryLifecyclePartialError) as exc_info:
        await mgr.add_memory_entry("u1", "x")

    outcome = exc_info.value.outcome
    assert outcome.operation is MemoryLifecycleOperation.WRITE
    assert outcome.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    assert outcome.primary_applied is True
    profile = await mgr.get_profile("u1")
    assert len(profile.memory_entries) == 1


@pytest.mark.asyncio
async def test_update_projection_failure_is_partial() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(fail_upsert=True)
    mgr = _manager(store, projection)

    with pytest.raises(UserProfileMemoryLifecyclePartialError):
        await mgr.update_memory_entry("u1", "e1", content="v2")

    profile = await mgr.get_profile("u1")
    assert profile.memory_entries[0].content == "v2"


@pytest.mark.asyncio
async def test_clear_memory_deletes_projections() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    await mgr.clear_memory("u1")

    assert ("e1",) in projection.delete_calls or ("e1",) == projection.delete_calls[-1]
    assert "e1" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_delete_profile_deletes_projections() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    await mgr.delete_profile("u1")

    assert "e1" in projection.delete_calls[0]
    assert "e1" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_delete_projection_failure_is_partial() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(fail_delete=True, indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    with pytest.raises(UserProfileMemoryLifecyclePartialError):
        await mgr.clear_memory("u1")


@pytest.mark.asyncio
async def test_reconcile_recreates_missing_projection() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection()
    mgr = _manager(store, projection)

    outcome = await mgr.reconcile_memory_projections("u1")

    assert outcome.disposition is MemoryReconciliationDisposition.REPAIRED
    assert ("u1", "e1") in projection.upsert_calls


@pytest.mark.asyncio
async def test_reconcile_removes_orphan_projection() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1", "orphan"})
    mgr = _manager(store, projection)

    outcome = await mgr.reconcile_memory_projections("u1")

    assert outcome.disposition is MemoryReconciliationDisposition.REPAIRED
    assert "orphan" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_reconcile_after_clear_removes_orphans() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)
    projection.fail_delete = True
    with pytest.raises(UserProfileMemoryLifecyclePartialError):
        await mgr.clear_memory("u1")
    projection.fail_delete = False

    outcome = await mgr.reconcile_memory_projections("u1")

    assert outcome.disposition is MemoryReconciliationDisposition.REPAIRED
    assert projection.indexed_entry_ids == set()


@pytest.mark.asyncio
async def test_projection_delete_is_idempotent() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    await mgr.remove_memory_entry("u1", "e1")
    await mgr.remove_memory_entry("u1", "e1")

    assert projection.delete_calls.count(("e1",)) >= 1
