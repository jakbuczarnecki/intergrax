# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-3: canonical Memory Control Plane contract and default implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.agent_run_enums import PrincipalType
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlForgetRequest,
    MemoryControlPlane,
    MemoryControlPlaneScope,
    MemoryControlRecallRequest,
    MemoryControlReconcileRequest,
    MemoryControlRememberRequest,
    MemoryControlScopeRef,
    MemoryControlUnsupportedScope,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    MemoryReconciliationDisposition,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
    user_profile_memory_projection_context,
)
from intergrax.memory.default_memory_control_plane import (
    DefaultMemoryControlPlane,
    UserProfileManagerMemoryCapability,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import (
    MemoryKind,
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)

pytestmark = pytest.mark.gate

_TENANT_A = "tenant-a"
_USER_U1 = "user-u1"
_PROJECTION_ID = "mem_ent_3_projection"


@dataclass
class RecordingMemoryProjection:
    projection_id: str = _PROJECTION_ID
    upsert_calls: list[tuple[str, str]] = field(default_factory=list)
    delete_calls: list[tuple[str, ...]] = field(default_factory=list)
    indexed_entry_ids: set[str] = field(default_factory=set)

    async def upsert_memory_entry(
        self,
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        self.upsert_calls.append((context.user_id, entry.entry_id))
        self.indexed_entry_ids.add(entry.entry_id)

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        self.delete_calls.append(tuple(entry_ids))
        for entry_id in entry_ids:
            self.indexed_entry_ids.discard(entry_id)

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        expected = set(context.authoritative_active_entry_ids)
        orphans = self.indexed_entry_ids - expected
        changed = bool(orphans)
        if orphans:
            await self.delete_memory_entries(
                user_profile_memory_projection_context(context.identity),
                tuple(sorted(orphans)),
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


def _identity(
    *,
    tenant_id: str = _TENANT_A,
    user_id: str = _USER_U1,
) -> RequestIdentity:
    return RequestIdentity(
        tenant_id=tenant_id,
        user_id=user_id,
        principal_type=PrincipalType.USER,
        auth_subject=user_id,
    )


def _plane(projection: RecordingMemoryProjection | None = None) -> DefaultMemoryControlPlane:
    store = InMemoryUserProfileStore()
    projections = (projection,) if projection is not None else ()
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT_A,
        memory_projections=projections,
    )
    return DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )


@pytest.mark.asyncio
async def test_remember_user_memory_runs_lifecycle_projection() -> None:
    projection = RecordingMemoryProjection()
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)

    result = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="likes python"),
    )

    assert result.scope is MemoryControlPlaneScope.USER
    assert result.entry_id
    assert projection.upsert_calls == [(_USER_U1, result.entry_id)]


@pytest.mark.asyncio
async def test_recall_user_memory_returns_canonical_entries() -> None:
    projection = RecordingMemoryProjection()
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)

    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="alpha fact"),
    )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="alpha", top_k=5),
    )

    assert recall.scope is MemoryControlPlaneScope.USER
    assert len(recall.items) == 1
    item = recall.items[0]
    assert item.entry_id == remembered.entry_id
    assert item.content == "alpha fact"
    assert item.kind is MemoryKind.OTHER


@pytest.mark.asyncio
async def test_forget_user_memory_clears_primary_and_projection() -> None:
    projection = RecordingMemoryProjection()
    plane = _plane(projection)
    identity = _identity()
    scope = user_memory_scope(identity)

    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="to delete"),
    )
    assert remembered.entry_id in projection.indexed_entry_ids

    await plane.forget(
        identity,
        scope,
        MemoryControlForgetRequest(entry_id=remembered.entry_id or ""),
    )

    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=10))
    assert all(item.entry_id != remembered.entry_id for item in recall.items)
    assert remembered.entry_id not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_reconcile_delegates_to_mem_ent_2_lifecycle() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection(indexed_entry_ids={"orphan-entry"})
    manager = UserProfileManager(
        store,
        tenant_id=_TENANT_A,
        memory_projections=(projection,),
    )
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER_U1),
        preferences=UserPreferences(),
        memory_entries=[
            UserProfileMemoryEntry(entry_id="active-entry", content="kept"),
        ],
    )
    await store.save_profile(tenant_id=_TENANT_A, profile=profile)
    plane = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(_manager=manager),
    )
    identity = _identity()
    scope = user_memory_scope(identity)

    result = await plane.reconcile(identity, scope, MemoryControlReconcileRequest())

    assert result.reconciliation is not None
    assert result.reconciliation.disposition is MemoryReconciliationDisposition.REPAIRED
    assert "orphan-entry" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_cross_tenant_scope_rejected() -> None:
    plane = _plane()
    identity = _identity(tenant_id=_TENANT_A)
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id="tenant-b",
        user_id=_USER_U1,
    )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="x"),
        )


@pytest.mark.asyncio
async def test_cross_user_scope_rejected() -> None:
    plane = _plane()
    identity = _identity(user_id=_USER_U1)
    scope = MemoryControlScopeRef(
        kind=MemoryControlPlaneScope.USER,
        tenant_id=_TENANT_A,
        user_id="other-user",
    )
    with pytest.raises(MemoryControlAccessDenied):
        await plane.forget(
            identity,
            scope,
            MemoryControlForgetRequest(entry_id="e1"),
        )


@pytest.mark.asyncio
async def test_unconfigured_user_capability_is_explicit() -> None:
    plane = DefaultMemoryControlPlane()
    identity = _identity()
    scope = user_memory_scope(identity)
    with pytest.raises(MemoryControlUnsupportedScope):
        await plane.remember(identity, scope, MemoryControlRememberRequest(content="x"))


@dataclass
class _FakeMemoryControlPlane:
    """Plugin replacement proving callers depend on the contract only."""

    remembered: list[str] = field(default_factory=list)

    async def remember(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRememberRequest,
    ) -> object:
        self.remembered.append(request.content)
        from intergrax.memory.contracts.memory_control import MemoryControlRememberResult

        return MemoryControlRememberResult(
            scope=scope.kind,
            entry_id="fake-entry",
        )

    async def recall(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlRecallRequest,
    ) -> object:
        from intergrax.memory.contracts.memory_control import MemoryControlRecallResult

        return MemoryControlRecallResult(scope=scope.kind, items=())

    async def forget(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlForgetRequest,
    ) -> object:
        from intergrax.memory.contracts.memory_control import MemoryControlForgetResult

        return MemoryControlForgetResult(scope=scope.kind)

    async def apply_memory_supersession(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        intent: object,
    ) -> object:
        from intergrax.memory.contracts.memory_control import MemoryControlSupersessionApplyResult
        from intergrax.memory.contracts.memory_lifecycle import (
            MemoryLifecycleDisposition,
            MemoryLifecycleOperation,
            MemoryLifecycleOutcome,
        )

        return MemoryControlSupersessionApplyResult(
            scope=scope.kind,
            superseded_memory_id="",
            superseding_memory_id="",
            lifecycle=MemoryLifecycleOutcome(
                operation=MemoryLifecycleOperation.UPDATE,
                disposition=MemoryLifecycleDisposition.COMPLETE,
                user_id="",
                memory_entity_ids=(),
                primary_applied=True,
                projection_evidence=(),
            ),
        )

    async def reconcile(
        self,
        identity: RequestIdentity,
        scope: MemoryControlScopeRef,
        request: MemoryControlReconcileRequest,
    ) -> object:
        from intergrax.memory.contracts.memory_control import MemoryControlReconcileResult

        return MemoryControlReconcileResult(scope=scope.kind)


@pytest.mark.asyncio
async def test_plugin_control_plane_is_swappable() -> None:
    fake = _FakeMemoryControlPlane()
    assert isinstance(fake, MemoryControlPlane)
    identity = _identity()
    scope = user_memory_scope(identity)
    await fake.remember(identity, scope, MemoryControlRememberRequest(content="plugin"))
    assert fake.remembered == ["plugin"]
