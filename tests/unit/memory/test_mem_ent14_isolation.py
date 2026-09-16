# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-14: provider failure isolation and observability failure isolation."""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityTypeRef,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlBackendError,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_observability import MemoryDiagnosticEvent
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_session_turn_index_store import (
    InMemorySessionTurnIndexStore,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_store import UserProfileStore
from tests.unit.memory.resilience.concurrency import run_two_async_tasks_with_barrier
from tests.unit.memory.resilience.fault_injection import YieldOnSaveUserProfileStore

pytestmark = pytest.mark.unit

_TENANT = "tenant-ent14-iso"
_USER = "user-ent14-iso"


class _TenantScopedFailingUserProfileStore(InMemoryUserProfileStore):
    """Fails only for tenant ``fail_tenant``; other tenants use normal behavior."""

    fail_tenant: str = "tenant-fail"

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        if tenant_id == self.fail_tenant:
            raise RuntimeError("provider failure for scoped tenant")
        await super().save_profile(tenant_id=tenant_id, profile=profile)


class _ConcurrentYieldFailingUserProfileStore(YieldOnSaveUserProfileStore):
    fail_tenant: str = "tenant-fail"

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        if tenant_id == self.fail_tenant:
            barrier = self.save_barrier
            if barrier is not None:
                await barrier.wait()
            raise RuntimeError("provider failure for scoped tenant")
        await super().save_profile(tenant_id=tenant_id, profile=profile)


class _FailingEntityStore(InMemoryEntityTemporalMemoryStore):
    fail_tenant: str = "tenant-fail"

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        if scope.tenant_id == self.fail_tenant:
            raise TimeoutError("entity provider failure")
        return super().upsert_entity(scope, record)


def _identity(tenant: str, user: str = _USER) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant,
        user_id=user,
        principal_type=PrincipalType.USER,
        auth_subject=user,
    )


@pytest.mark.asyncio
async def test_user_profile_tenant_isolation_under_real_async_task_concurrency() -> None:
    store = YieldOnSaveUserProfileStore()
    store.save_barrier = asyncio.Barrier(2)
    profile_a = UserProfile(
        identity=UserIdentity(user_id="user-a"),
        preferences=UserPreferences(),
        system_instructions="tenant-a-marker",
    )
    profile_b = UserProfile(
        identity=UserIdentity(user_id="user-b"),
        preferences=UserPreferences(),
        system_instructions="tenant-b-marker",
    )

    async def _save_a() -> None:
        await store.save_profile(tenant_id="tenant-a", profile=profile_a)

    async def _save_b() -> None:
        await store.save_profile(tenant_id="tenant-b", profile=profile_b)

    await run_two_async_tasks_with_barrier(
        start_barrier=asyncio.Barrier(2),
        task_a=_save_a,
        task_b=_save_b,
    )
    loaded_a = await store.get_profile(tenant_id="tenant-a", user_id="user-a")
    loaded_b = await store.get_profile(tenant_id="tenant-b", user_id="user-b")
    assert loaded_a.system_instructions == "tenant-a-marker"
    assert loaded_b.system_instructions == "tenant-b-marker"


@pytest.mark.asyncio
async def test_user_profile_provider_failure_does_not_corrupt_other_tenant() -> None:
    store = _TenantScopedFailingUserProfileStore()
    good_profile = UserProfile(
        identity=UserIdentity(user_id="user-good"),
        preferences=UserPreferences(),
    )
    await store.save_profile(tenant_id="tenant-ok", profile=good_profile)
    with pytest.raises(RuntimeError, match="provider failure"):
        await store.save_profile(
            tenant_id="tenant-fail",
            profile=UserProfile(
                identity=UserIdentity(user_id="user-bad"),
                preferences=UserPreferences(),
            ),
        )
    loaded = await store.get_profile(tenant_id="tenant-ok", user_id="user-good")
    assert loaded.identity.user_id == "user-good"


def test_entity_provider_failure_isolated_per_tenant() -> None:
    store = _FailingEntityStore()
    ok_scope = EntityMemoryScope(tenant_id="tenant-ok", user_id=_USER)
    fail_scope = EntityMemoryScope(tenant_id="tenant-fail", user_id=_USER)
    record = EntityRecord(
        entity_id="ent-1",
        entity_type=EntityTypeRef("person"),
        canonical_name="ok",
        revision=1,
        created_at="2025-01-01T00:00:00+00:00",
    )
    store.upsert_entity(ok_scope, record)
    with pytest.raises(TimeoutError):
        store.upsert_entity(fail_scope, record)
    assert store.get_entity(ok_scope, "ent-1") is not None


class _YieldOnUpsertSessionTurnIndexStore(InMemorySessionTurnIndexStore):
    upsert_barrier: asyncio.Barrier | None = None

    async def upsert_turn(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str | None,
        message: ChatMessage,
    ) -> None:
        barrier = self.upsert_barrier
        if barrier is not None:
            await barrier.wait()
        await super().upsert_turn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id,
            message=message,
        )


@pytest.mark.asyncio
async def test_session_turn_index_tenant_isolation_under_real_async_task_concurrency() -> None:
    store = _YieldOnUpsertSessionTurnIndexStore()
    store.upsert_barrier = asyncio.Barrier(2)
    msg_a = ChatMessage(role="user", content="tenant A turn", entry_id="turn-a")
    msg_b = ChatMessage(role="user", content="tenant B turn", entry_id="turn-b")

    async def _upsert_a() -> None:
        await store.upsert_turn(
            tenant_id="tenant-a",
            session_id="sess-a",
            user_id="user-a",
            message=msg_a,
        )

    async def _upsert_b() -> None:
        await store.upsert_turn(
            tenant_id="tenant-b",
            session_id="sess-b",
            user_id="user-b",
            message=msg_b,
        )

    await run_two_async_tasks_with_barrier(
        start_barrier=asyncio.Barrier(2),
        task_a=_upsert_a,
        task_b=_upsert_b,
    )
    hits_a = await store.search_turns(query="tenant", tenant_id="tenant-a", top_k=5)
    hits_b = await store.search_turns(query="tenant", tenant_id="tenant-b", top_k=5)
    assert len(hits_a) == 1
    assert len(hits_b) == 1
    assert hits_a[0].tenant_id == "tenant-a"
    assert hits_b[0].tenant_id == "tenant-b"


@pytest.mark.asyncio
async def test_session_turn_index_tenant_isolation_sequential_upserts() -> None:
    store = InMemorySessionTurnIndexStore()
    msg_a = ChatMessage(role="user", content="tenant A turn", entry_id="turn-a")
    msg_b = ChatMessage(role="user", content="tenant B turn", entry_id="turn-b")
    await store.upsert_turn(
        tenant_id="tenant-a",
        session_id="sess-a",
        user_id="user-a",
        message=msg_a,
    )
    await store.upsert_turn(
        tenant_id="tenant-b",
        session_id="sess-b",
        user_id="user-b",
        message=msg_b,
    )
    hits_a = await store.search_turns(query="tenant", tenant_id="tenant-a", top_k=5)
    hits_b = await store.search_turns(query="tenant", tenant_id="tenant-b", top_k=5)
    assert len(hits_a) == 1
    assert len(hits_b) == 1
    assert hits_a[0].tenant_id == "tenant-a"
    assert hits_b[0].tenant_id == "tenant-b"


@pytest.mark.asyncio
async def test_user_profile_provider_failure_concurrent_with_successful_tenant() -> None:
    store = _ConcurrentYieldFailingUserProfileStore()
    store.save_barrier = asyncio.Barrier(2)
    good_profile = UserProfile(
        identity=UserIdentity(user_id="user-good"),
        preferences=UserPreferences(),
        system_instructions="ok-marker",
    )

    async def _save_good() -> None:
        await store.save_profile(tenant_id="tenant-ok", profile=good_profile)

    async def _save_bad() -> None:
        with pytest.raises(RuntimeError, match="provider failure"):
            await store.save_profile(
                tenant_id="tenant-fail",
                profile=UserProfile(
                    identity=UserIdentity(user_id="user-bad"),
                    preferences=UserPreferences(),
                ),
            )

    await run_two_async_tasks_with_barrier(
        start_barrier=asyncio.Barrier(2),
        task_a=_save_good,
        task_b=_save_bad,
    )
    loaded = await store.get_profile(tenant_id="tenant-ok", user_id="user-good")
    assert loaded.system_instructions == "ok-marker"


@pytest.mark.asyncio
async def test_provider_failure_with_exploding_sink_preserves_primary_error() -> None:
    class _ExplodingSink:
        def record(self, event: MemoryDiagnosticEvent) -> None:
            raise RuntimeError("sink down")

    class _AlwaysFailingStore(UserProfileStore):
        async def get_profile(self, *, tenant_id: str, user_id: str) -> UserProfile:
            raise RuntimeError("backend read failed")

        async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
            raise RuntimeError("backend write failed")

        async def delete_profile(self, *, tenant_id: str, user_id: str) -> None:
            raise RuntimeError("backend delete failed")

    emitter = MemoryDiagnosticEmitter(_sink=_ExplodingSink())
    mgr = UserProfileManager(
        _AlwaysFailingStore(),
        tenant_id=_TENANT,
        diagnostic_emitter=emitter,
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=mgr),
        diagnostic_emitter=emitter,
    )
    identity = _identity(_TENANT)
    scope = user_memory_scope(identity)
    with pytest.raises(MemoryControlBackendError):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="should fail"),
        )
