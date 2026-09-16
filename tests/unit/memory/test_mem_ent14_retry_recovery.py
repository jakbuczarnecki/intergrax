# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-14: retry idempotency, partial failure, reconciliation, durable reopen."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from typing import Sequence

import pytest

from intergrax.contracts.agent_run import PrincipalType, RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryProjectionFailureCategory,
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    MemoryReconciliationDisposition,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.stores.sqlite_user_profile_store import SQLiteUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from tests.unit.memory.resilience.fault_injection import (
    AmbiguousCommitUserProfileStore,
    FailAfterCommitOnceUserProfileStore,
    FailBeforeCommitUserProfileStore,
    FailFirstThenSucceedUserProfileStore,
)
from tests.unit.memory.resilience.recovery_projection import FailOnceRepairableProjection
from tests.unit.memory.test_mem_ent8_procedural_memory import _procedure

pytestmark = pytest.mark.unit

_TENANT = "tenant-ent14-retry"
_USER = "user-ent14-retry"


def _identity() -> RequestIdentity:
    return RequestIdentity(
        tenant_id=_TENANT,
        user_id=_USER,
        principal_type=PrincipalType.USER,
        auth_subject=_USER,
    )


@pytest.mark.asyncio
async def test_ambiguous_commit_retry_preserves_single_semantic_profile_state() -> None:
    """Multiple physical writes after ambiguous failure → one semantic final state."""
    store = AmbiguousCommitUserProfileStore()
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER),
        preferences=UserPreferences(preferred_language="pl"),
        system_instructions="marker-ambiguous",
    )
    with pytest.raises(TimeoutError):
        await store.save_profile(tenant_id=_TENANT, profile=profile)
    with pytest.raises(TimeoutError):
        await store.save_profile(tenant_id=_TENANT, profile=profile)
    loaded = await store.get_profile(tenant_id=_TENANT, user_id=_USER)
    assert loaded.system_instructions == "marker-ambiguous"
    assert store.commit_count == 2


@pytest.mark.asyncio
async def test_fail_before_commit_real_second_attempt_persists_once() -> None:
    store = FailFirstThenSucceedUserProfileStore()
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER),
        preferences=UserPreferences(),
        system_instructions="persist-on-second-attempt",
    )
    with pytest.raises(TimeoutError):
        await store.save_profile(tenant_id=_TENANT, profile=profile)
    await store.save_profile(tenant_id=_TENANT, profile=profile)
    loaded = await store.get_profile(tenant_id=_TENANT, user_id=_USER)
    assert loaded.system_instructions == "persist-on-second-attempt"
    assert store.attempts == 2


@pytest.mark.asyncio
async def test_fail_before_commit_store_without_retry_leaves_empty_profile() -> None:
    store = FailBeforeCommitUserProfileStore()
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER),
        preferences=UserPreferences(),
        system_instructions="never-persisted",
    )
    with pytest.raises(TimeoutError):
        await store.save_profile(tenant_id=_TENANT, profile=profile)
    loaded = await store.get_profile(tenant_id=_TENANT, user_id=_USER)
    assert (loaded.system_instructions or "") == ""
    assert store.attempts == 1


@pytest.mark.asyncio
async def test_fail_after_commit_once_second_call_succeeds_with_single_semantic_state() -> None:
    store = FailAfterCommitOnceUserProfileStore()
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER),
        preferences=UserPreferences(),
        system_instructions="ambiguous-once",
    )
    with pytest.raises(TimeoutError):
        await store.save_profile(tenant_id=_TENANT, profile=profile)
    await store.save_profile(tenant_id=_TENANT, profile=profile)
    loaded = await store.get_profile(tenant_id=_TENANT, user_id=_USER)
    assert loaded.system_instructions == "ambiguous-once"
    assert store.commit_count == 2


@pytest.mark.asyncio
async def test_repeated_procedural_delete_is_safe() -> None:
    from intergrax.memory.contracts.procedural_memory import ProceduralMemoryScope

    from intergrax.memory.contracts.procedural_memory import procedure_id_for_source_memory

    store = InMemoryProceduralMemoryStore()
    scope = ProceduralMemoryScope(tenant_id=_TENANT, user_id=_USER)
    pid = procedure_id_for_source_memory(scope, "mem-del")
    store.upsert_procedure(scope, _procedure(pid, source_memory_id="mem-del"))
    assert store.delete_by_source_memory(scope, "mem-del") == 1
    assert store.delete_by_source_memory(scope, "mem-del") == 0
    assert store.delete_by_source_memory(scope, "mem-del") == 0


@dataclass
class _RecordingProjection:
    projection_id: str = "recording"
    upserts: list[str] = field(default_factory=list)
    reconcile_calls: int = 0

    async def upsert_memory_entry(
        self,
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.upserts.append(entry.entry_id)

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        return None

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        self.reconcile_calls += 1
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
        )


@pytest.mark.asyncio
async def test_reconciliation_repeated_execution_is_idempotent() -> None:
    store = InMemoryUserProfileStore()
    await store.save_profile(
        tenant_id=_TENANT,
        profile=UserProfile(
            identity=UserIdentity(user_id=_USER),
            preferences=UserPreferences(),
            memory_entries=[
                UserProfileMemoryEntry(
                    entry_id="e1",
                    content="fact",
                    kind=MemoryKind.USER_FACT,
                    revision=1,
                ),
            ],
        ),
    )
    projection = _RecordingProjection()
    mgr = UserProfileManager(
        store,
        tenant_id=_TENANT,
        memory_projections=(projection,),
    )
    first = await mgr.reconcile_memory_projections(_identity())
    second = await mgr.reconcile_memory_projections(_identity())
    assert first.disposition is MemoryReconciliationDisposition.CONSISTENT
    assert second.disposition is MemoryReconciliationDisposition.CONSISTENT
    assert projection.reconcile_calls == 2


@pytest.mark.asyncio
async def test_projection_failure_after_canonical_remember_reports_partial_failure() -> None:
    @dataclass
    class _FailingProjection:
        projection_id: str = "fail-upsert"

        async def upsert_memory_entry(
            self,
            context: UserProfileMemoryProjectionContext,
            entry: UserProfileMemoryEntry,
        ) -> None:
            raise TimeoutError("projection failed")

        async def delete_memory_entries(
            self,
            context: UserProfileMemoryProjectionContext,
            entry_ids: Sequence[str],
        ) -> None:
            return None

        async def reconcile(
            self,
            context: UserProfileMemoryReconciliationContext,
        ) -> MemoryProjectionReconciliationResult:
            return MemoryProjectionReconciliationResult(
                projection_id=self.projection_id,
                disposition=MemoryProjectionReconciliationDisposition.CONSISTENT,
            )

    store = InMemoryUserProfileStore()
    mgr = UserProfileManager(
        store,
        tenant_id=_TENANT,
        memory_projections=(_FailingProjection(),),
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=mgr),
    )
    identity = _identity()
    scope = user_memory_scope(identity)
    from intergrax.memory.contracts.memory_control import MemoryControlPartialLifecycleError

    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="canonical fact"),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE
    assert lifecycle.primary_applied is True
    profile = await store.get_profile(tenant_id=_TENANT, user_id=_USER)
    assert profile.memory_entries
    failed = [ev for ev in lifecycle.projection_evidence if not ev.succeeded]
    assert failed
    assert failed[0].failure is not None
    assert failed[0].failure.category is MemoryProjectionFailureCategory.RETRYABLE


@pytest.mark.asyncio
async def test_projection_failure_then_reconcile_repairs_missing_entry() -> None:
    store = InMemoryUserProfileStore()
    projection = FailOnceRepairableProjection()
    mgr = UserProfileManager(
        store,
        tenant_id=_TENANT,
        memory_projections=(projection,),
    )
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=mgr),
    )
    identity = _identity()
    scope = user_memory_scope(identity)
    from intergrax.memory.contracts.memory_control import MemoryControlPartialLifecycleError

    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="canonical fact for repair"),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.primary_applied is True
    assert projection.entries == {}
    assert projection.upsert_attempts == 1

    first = await mgr.reconcile_memory_projections(identity)
    assert first.disposition is MemoryReconciliationDisposition.REPAIRED
    assert projection.repair_count == 1
    assert len(projection.entries) == 1

    repair_before = projection.repair_count
    second = await mgr.reconcile_memory_projections(identity)
    assert second.disposition is MemoryReconciliationDisposition.CONSISTENT
    assert projection.repair_count == repair_before


@pytest.mark.asyncio
async def test_sqlite_user_profile_reopen_then_continue_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/profiles.db"
        marker_v1 = "durability-v1"
        marker_v2 = "durability-v2"

        writer = SQLiteUserProfileStore(db_path)
        await writer.save_profile(
            tenant_id=_TENANT,
            profile=UserProfile(
                identity=UserIdentity(user_id=_USER),
                preferences=UserPreferences(preferred_language="en"),
                system_instructions=marker_v1,
            ),
        )
        writer.close()

        reader = SQLiteUserProfileStore(db_path)
        loaded = await reader.get_profile(tenant_id=_TENANT, user_id=_USER)
        assert loaded.system_instructions == marker_v1
        await reader.save_profile(
            tenant_id=_TENANT,
            profile=UserProfile(
                identity=UserIdentity(user_id=_USER),
                preferences=UserPreferences(preferred_language="de"),
                system_instructions=marker_v2,
            ),
        )
        reader.close()

        reopened = SQLiteUserProfileStore(db_path)
        final = await reopened.get_profile(tenant_id=_TENANT, user_id=_USER)
        reopened.close()
        assert final.system_instructions == marker_v2
        assert final.preferences.preferred_language == "de"


@pytest.mark.asyncio
async def test_sqlite_cross_tenant_isolation_after_reopen() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/profiles-iso.db"
        store = SQLiteUserProfileStore(db_path)
        await store.save_profile(
            tenant_id="tenant-a",
            profile=UserProfile(
                identity=UserIdentity(user_id=_USER),
                preferences=UserPreferences(),
                system_instructions="tenant-a-marker",
            ),
        )
        store.close()

        reopened = SQLiteUserProfileStore(db_path)
        await reopened.save_profile(
            tenant_id="tenant-b",
            profile=UserProfile(
                identity=UserIdentity(user_id=_USER),
                preferences=UserPreferences(),
                system_instructions="tenant-b-marker",
            ),
        )
        a = await reopened.get_profile(tenant_id="tenant-a", user_id=_USER)
        b = await reopened.get_profile(tenant_id="tenant-b", user_id=_USER)
        reopened.close()
        assert a.system_instructions == "tenant-a-marker"
        assert b.system_instructions == "tenant-b-marker"
