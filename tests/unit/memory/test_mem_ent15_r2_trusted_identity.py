# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-15-R2: trusted RequestIdentity authority in user-scoped mutations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_lifecycle import (
    UserProfileMemoryProjection,
    UserProfileMemoryProjectionContext,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry

pytestmark = pytest.mark.gate


@dataclass
class _CountingProjection:
    upserts: int = 0
    projection_id: str = "counting"

    async def upsert_memory_entry(
        self,
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.upserts += 1

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: tuple[str, ...] | list[str],
    ) -> None:
        return None

    async def reconcile(self, context: object) -> object:
        from intergrax.memory.contracts.memory_lifecycle import (
            MemoryProjectionReconciliationDisposition,
            MemoryProjectionReconciliationResult,
        )

        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
        )


@pytest.mark.asyncio
async def test_manager_rejects_missing_identity_user_id_before_projection() -> None:
    projection = _CountingProjection()
    mgr = UserProfileManager(
        InMemoryUserProfileStore(),
        tenant_id="tenant-a",
        memory_projections=(projection,),
    )
    identity = RequestIdentity(tenant_id="tenant-a", user_id=None)
    with pytest.raises(ValueError, match="user_id is required"):
        await mgr.add_memory_entry_with_lifecycle(identity, "user-a", "fact")
    assert projection.upserts == 0


@pytest.mark.asyncio
async def test_manager_rejects_identity_user_mismatch_before_projection() -> None:
    projection = _CountingProjection()
    mgr = UserProfileManager(
        InMemoryUserProfileStore(),
        tenant_id="tenant-a",
        memory_projections=(projection,),
    )
    identity = RequestIdentity(tenant_id="tenant-a", user_id="user-a")
    with pytest.raises(ValueError, match="conflicts"):
        await mgr.add_memory_entry_with_lifecycle(identity, "user-b", "fact")
    assert projection.upserts == 0
