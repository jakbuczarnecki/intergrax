# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryLifecycleOperation,
    MemoryProjectionFailureCategory,
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    MemoryReconciliationDisposition,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
    user_profile_memory_projection_context,
)
from tests.unit.memory._projection_identity import memory_test_identity
from intergrax.memory.user_profile_memory_lifecycle import UserProfileMemoryLifecycleCoordinator
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
_U1 = memory_test_identity(tenant_id="tenant-a", user_id="u1")


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
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        if self.fail_upsert:
            raise TimeoutError("projection unavailable")
        self.upsert_calls.append((context.user_id, entry.entry_id))
        self.indexed_entry_ids.add(entry.entry_id)

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        if self.fail_delete:
            raise TimeoutError("projection unavailable")
        self.delete_calls.append(tuple(entry_ids))
        for entry_id in entry_ids:
            self.indexed_entry_ids.discard(entry_id)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        self.reconcile_calls += 1
        expected = set(context.authoritative_active_entry_ids)
        orphans = self.indexed_entry_ids - expected
        changed = bool(orphans)
        if orphans:
            await self.delete_memory_entries(
                user_profile_memory_projection_context(context.identity),
                tuple(sorted(orphans)),
            )
        if context.profile is None:
            if self.indexed_entry_ids:
                changed = True
                await self.delete_memory_entries(
                    user_profile_memory_projection_context(context.identity),
                    tuple(sorted(self.indexed_entry_ids)),
                )
        else:
            for entry in context.profile.memory_entries:
                if entry.deleted:
                    continue
                if entry.entry_id not in self.indexed_entry_ids:
                    changed = True
                    await self.upsert_memory_entry(
                        user_profile_memory_projection_context(context.identity),
                        entry,
                    )
        disposition = (
            MemoryProjectionReconciliationDisposition.REPAIRED
            if changed
            else MemoryProjectionReconciliationDisposition.CONSISTENT
        )
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=disposition,
        )


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

    entry = await mgr.add_memory_entry(_U1, "u1", "remember this")

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
        await mgr.add_memory_entry(_U1, "u1", "x")

    assert projection.upsert_calls == []


@pytest.mark.asyncio
async def test_write_projection_failure_leaves_primary_and_raises_partial() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection(fail_upsert=True)
    mgr = _manager(store, projection)

    with pytest.raises(UserProfileMemoryLifecyclePartialError) as exc_info:
        await mgr.add_memory_entry(_U1, "u1", "x")

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
        await mgr.update_memory_entry(_U1, "u1", "e1", content="v2")

    profile = await mgr.get_profile("u1")
    assert profile.memory_entries[0].content == "v2"


@pytest.mark.asyncio
async def test_clear_memory_deletes_projections() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    await mgr.clear_memory(_U1, "u1")

    assert ("e1",) in projection.delete_calls or ("e1",) == projection.delete_calls[-1]
    assert "e1" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_delete_profile_deletes_projections() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    await mgr.delete_profile(_U1, "u1")

    assert "e1" in projection.delete_calls[0]
    assert "e1" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_delete_projection_failure_is_partial() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(fail_delete=True, indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    with pytest.raises(UserProfileMemoryLifecyclePartialError):
        await mgr.clear_memory(_U1, "u1")


@pytest.mark.asyncio
async def test_reconcile_recreates_missing_projection() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection()
    mgr = _manager(store, projection)

    outcome = await mgr.reconcile_memory_projections(_U1)

    assert outcome.disposition is MemoryReconciliationDisposition.REPAIRED
    assert ("u1", "e1") in projection.upsert_calls


@pytest.mark.asyncio
async def test_reconcile_removes_orphan_projection() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1", "orphan"})
    mgr = _manager(store, projection)

    outcome = await mgr.reconcile_memory_projections(_U1)

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
        await mgr.clear_memory(_U1, "u1")
    projection.fail_delete = False

    outcome = await mgr.reconcile_memory_projections(_U1)

    assert outcome.disposition is MemoryReconciliationDisposition.REPAIRED
    assert projection.indexed_entry_ids == set()


@pytest.mark.asyncio
async def test_reconcile_consistent_when_projection_already_consistent() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    outcome = await mgr.reconcile_memory_projections(_U1)

    assert outcome.disposition is MemoryReconciliationDisposition.CONSISTENT
    assert projection.upsert_calls == []


@pytest.mark.asyncio
async def test_reconcile_coordinator_failed_when_one_projection_raises() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    ok_projection = RecordingMemoryProjection(
        projection_id="ok",
        indexed_entry_ids={"e1"},
    )

    class FailingProjection:
        projection_id = "failing"

        async def upsert_memory_entry(
            self,
            context: UserProfileMemoryProjectionContext,
            entry: UserProfileMemoryEntry,
        ) -> None:
            raise AssertionError("not used")

        async def delete_memory_entries(
            self,
            context: UserProfileMemoryProjectionContext,
            entry_ids: Sequence[str],
        ) -> None:
            raise AssertionError("not used")

        async def reconcile(
            self,
            context: UserProfileMemoryReconciliationContext,
        ) -> MemoryProjectionReconciliationResult:
            raise TimeoutError("projection read failed")

    coordinator = UserProfileMemoryLifecycleCoordinator(
        projections=(ok_projection, FailingProjection()),
    )
    profile = await store.get_profile(tenant_id="tenant-a", user_id="u1")
    outcome = await coordinator.reconcile_user(identity=_U1, profile=profile)

    assert outcome.disposition is MemoryReconciliationDisposition.FAILED
    failed = [item for item in outcome.projection_evidence if not item.succeeded]
    assert len(failed) == 1
    assert failed[0].failure is not None
    assert failed[0].failure.category is MemoryProjectionFailureCategory.RETRYABLE


@pytest.mark.asyncio
async def test_reconcile_idempotent_second_pass_consistent() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection()
    mgr = _manager(store, projection)

    first = await mgr.reconcile_memory_projections(_U1)
    second = await mgr.reconcile_memory_projections(_U1)

    assert first.disposition is MemoryReconciliationDisposition.REPAIRED
    assert second.disposition is MemoryReconciliationDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_projection_delete_is_idempotent() -> None:
    store = InMemoryUserProfileStore()
    await _seed_profile(store, "u1", "e1")
    projection = RecordingMemoryProjection(indexed_entry_ids={"e1"})
    mgr = _manager(store, projection)

    await mgr.remove_memory_entry(_U1, "u1", "e1")
    await mgr.remove_memory_entry(_U1, "u1", "e1")

    assert projection.delete_calls.count(("e1",)) >= 1
