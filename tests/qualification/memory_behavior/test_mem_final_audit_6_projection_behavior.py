# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — projection lifecycle and canonical authority gates."""

from __future__ import annotations

import pytest

from intergrax.memory.contracts.memory_control import (
    MemoryControlForgetRequest,
    MemoryControlPartialLifecycleError,
    MemoryControlRecallRequest,
    MemoryControlReconcileRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    MemoryReconciliationDisposition,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.user_profile_memory import (
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from tests.qualification.memory_behavior.contracts import BehaviorViolationLedger
from tests.qualification.memory_behavior.fixtures import (
    FailingDeleteLtmVectorProjection,
    RecordingMemoryProjection,
    TENANT_A,
    build_user_control_plane,
    request_identity,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from tests.qualification.memory_behavior.fixtures import FixedEmbeddingManager

pytestmark = pytest.mark.gate

_USER = "audit6-proj-user"


@pytest.mark.asyncio
async def test_p01_partial_projection_remember() -> None:
    projection = RecordingMemoryProjection(fail_upsert=True)
    plane, _ = build_user_control_plane(projection=projection)
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    with pytest.raises(MemoryControlPartialLifecycleError) as exc_info:
        await plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(content="partial write"),
        )
    lifecycle = exc_info.value.lifecycle
    assert lifecycle.primary_applied is True
    assert lifecycle.requires_reconciliation is True
    assert lifecycle.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE


@pytest.mark.asyncio
async def test_p02_partial_projection_forget() -> None:
    projection = RecordingMemoryProjection()
    plane, _ = build_user_control_plane(projection=projection)
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="to delete"),
    )
    projection.fail_delete = True
    with pytest.raises(MemoryControlPartialLifecycleError):
        await plane.forget(
            identity,
            scope,
            MemoryControlForgetRequest(entry_id=remembered.entry_id or ""),
        )
    recall = await plane.recall(identity, scope, MemoryControlRecallRequest(top_k=10))
    assert all(item.entry_id != remembered.entry_id for item in recall.items)


@pytest.mark.asyncio
async def test_p03_reconciliation_repair() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection(indexed_entry_ids={"orphan-entry"})
    manager = UserProfileManager(store, tenant_id=TENANT_A, memory_projections=(projection,))
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER),
        preferences=UserPreferences(),
        memory_entries=[UserProfileMemoryEntry(entry_id="active-entry", content="kept")],
    )
    await store.save_profile(tenant_id=TENANT_A, profile=profile)
    plane = build_default_memory_control_plane(user_profile_manager=manager)
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    result = await plane.reconcile(identity, scope, MemoryControlReconcileRequest())
    assert result.reconciliation is not None
    assert result.reconciliation.disposition is MemoryReconciliationDisposition.REPAIRED
    assert "orphan-entry" not in projection.indexed_entry_ids


@pytest.mark.asyncio
async def test_p04_reconciliation_idempotent() -> None:
    store = InMemoryUserProfileStore()
    projection = RecordingMemoryProjection(indexed_entry_ids={"orphan-entry"})
    manager = UserProfileManager(store, tenant_id=TENANT_A, memory_projections=(projection,))
    profile = UserProfile(
        identity=UserIdentity(user_id=_USER),
        preferences=UserPreferences(),
        memory_entries=[UserProfileMemoryEntry(entry_id="active-entry", content="kept")],
    )
    await store.save_profile(tenant_id=TENANT_A, profile=profile)
    plane = build_default_memory_control_plane(user_profile_manager=manager)
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    first = await plane.reconcile(identity, scope, MemoryControlReconcileRequest())
    second = await plane.reconcile(identity, scope, MemoryControlReconcileRequest())
    assert first.reconciliation is not None
    assert second.reconciliation is not None
    assert second.reconciliation.disposition is MemoryReconciliationDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_user_08_stale_vector_after_delete(violation_ledger: BehaviorViolationLedger) -> None:
    backend = InMemoryVectorStore(TENANT_A)
    vector_manager = VectorstoreManager(backend, scope=VectorStoreScope(tenant_id=TENANT_A))
    stale_ltm = FailingDeleteLtmVectorProjection(
        embedding_manager=FixedEmbeddingManager(),  # type: ignore[arg-type]
        vectorstore_manager=vector_manager,
        tenant_id=TENANT_A,
        vector_index_namespace=None,
        workspace_id=None,
    )
    plane, _ = build_user_control_plane(enable_ltm_vector=True, ltm_projection=stale_ltm)
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    remembered = await plane.remember(
        identity,
        scope,
        MemoryControlRememberRequest(content="Preferred language is Polish."),
    )
    assert remembered.entry_id
    with pytest.raises(MemoryControlPartialLifecycleError):
        await plane.forget(
            identity,
            scope,
            MemoryControlForgetRequest(entry_id=remembered.entry_id or ""),
        )
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(query="language Polish", top_k=5),
    )
    for item in recall.items:
        if item.entry_id == remembered.entry_id:
            violation_ledger.record_projection_ghost()
            violation_ledger.record_deleted_resurrection()
            raise AssertionError("canonical deleted entry must not appear in recall (projection ghost)")


class FailingReconcileProjection(RecordingMemoryProjection):
    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=MemoryProjectionReconciliationDisposition.FAILED,
        )


@pytest.mark.asyncio
async def test_p05_reconciliation_projection_failure() -> None:
    projection = FailingReconcileProjection()
    plane, _ = build_user_control_plane(projection=projection)
    identity = request_identity(user_id=_USER)
    scope = user_memory_scope(identity)
    await plane.remember(identity, scope, MemoryControlRememberRequest(content="x"))
    result = await plane.reconcile(identity, scope, MemoryControlReconcileRequest())
    assert result.reconciliation is not None
    assert result.reconciliation.disposition is MemoryReconciliationDisposition.FAILED
