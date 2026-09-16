# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionFailureCategory,
    MemoryProjectionReconciliationDisposition,
    MemoryReconciliationDisposition,
    UserProfileMemoryReconciliationContext,
    user_profile_memory_projection_context,
)
from tests.unit.memory._projection_identity import memory_test_identity
from intergrax.memory.memory_vector_namespace import LTM_INDEX_DOMAIN, resolve_memory_index_collection
from intergrax.memory.user_profile_ltm_vector_projection import UserProfileLtmVectorProjection
from intergrax.memory.user_profile_memory import (
    UserIdentity,
    UserPreferences,
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.memory.user_profile_memory_lifecycle import UserProfileMemoryLifecycleCoordinator
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.gate

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"
USER_ID = "user-1"
_RECONCILE_IDENTITY = memory_test_identity(tenant_id=TENANT_A, user_id=USER_ID)


class _FixedEmbeddingManager:
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _profile(entry_ids: Sequence[str]) -> UserProfile:
    return UserProfile(
        identity=UserIdentity(user_id=USER_ID),
        preferences=UserPreferences(),
        memory_entries=[
            UserProfileMemoryEntry(entry_id=entry_id, content=f"content-{entry_id}")
            for entry_id in entry_ids
        ],
    )


def _context(
    profile: UserProfile | None,
    active_ids: frozenset[str],
) -> UserProfileMemoryReconciliationContext:
    return UserProfileMemoryReconciliationContext(
        identity=_RECONCILE_IDENTITY,
        profile=profile,
        authoritative_active_entry_ids=active_ids,
    )


def _ltm_projection(
    *,
    tenant_id: str = TENANT_A,
    store: VectorStore | None = None,
) -> UserProfileLtmVectorProjection:
    backend = store or InMemoryVectorStore(tenant_id)
    manager = VectorstoreManager(backend, scope=VectorStoreScope(tenant_id=tenant_id))
    return UserProfileLtmVectorProjection(
        embedding_manager=_FixedEmbeddingManager(),  # type: ignore[arg-type]
        vectorstore_manager=manager,
        tenant_id=tenant_id,
        vector_index_namespace=None,
        workspace_id=None,
    )


def _collection_name(tenant_id: str) -> str:
    return resolve_memory_index_collection(
        vector_index_namespace=None,
        tenant_id=tenant_id,
        domain=LTM_INDEX_DOMAIN,
    )


async def _seed_vector(
    projection: UserProfileLtmVectorProjection,
    entry_id: str,
    *,
    tenant_id: str = TENANT_A,
) -> None:
    entry = UserProfileMemoryEntry(entry_id=entry_id, content=f"content-{entry_id}")
    await projection.upsert_memory_entry(
        user_profile_memory_projection_context(_RECONCILE_IDENTITY),
        entry,
    )


@pytest.mark.asyncio
async def test_ltm_reconcile_consistent_when_vector_matches_primary() -> None:
    projection = _ltm_projection()
    await _seed_vector(projection, "e1")
    profile = _profile(["e1"])

    result = await projection.reconcile(_context(profile, frozenset({"e1"})))

    assert result.disposition is MemoryProjectionReconciliationDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_ltm_reconcile_repairs_missing_vector_entry() -> None:
    projection = _ltm_projection()
    profile = _profile(["e1"])

    result = await projection.reconcile(_context(profile, frozenset({"e1"})))

    assert result.disposition is MemoryProjectionReconciliationDisposition.REPAIRED
    second = await projection.reconcile(_context(profile, frozenset({"e1"})))
    assert second.disposition is MemoryProjectionReconciliationDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_ltm_reconcile_removes_orphan_vector_entry() -> None:
    projection = _ltm_projection()
    await _seed_vector(projection, "e1")
    await _seed_vector(projection, "orphan")
    profile = _profile(["e1"])

    result = await projection.reconcile(_context(profile, frozenset({"e1"})))

    assert result.disposition is MemoryProjectionReconciliationDisposition.REPAIRED
    follow_up = await projection.reconcile(_context(profile, frozenset({"e1"})))
    assert follow_up.disposition is MemoryProjectionReconciliationDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_ltm_reconcile_deleted_profile_clears_vector() -> None:
    projection = _ltm_projection()
    await _seed_vector(projection, "orphan")
    profile = None

    result = await projection.reconcile(_context(profile, frozenset()))

    assert result.disposition is MemoryProjectionReconciliationDisposition.REPAIRED
    follow_up = await projection.reconcile(_context(profile, frozenset()))
    assert follow_up.disposition is MemoryProjectionReconciliationDisposition.CONSISTENT


@pytest.mark.asyncio
async def test_ltm_reconcile_scope_isolation_across_tenants() -> None:
    store_a = InMemoryVectorStore(TENANT_A)
    store_b = InMemoryVectorStore(TENANT_B)
    projection_a = _ltm_projection(tenant_id=TENANT_A, store=store_a)
    projection_b = _ltm_projection(tenant_id=TENANT_B, store=store_b)
    await _seed_vector(projection_a, "e1", tenant_id=TENANT_A)
    await _seed_vector(projection_b, "e2", tenant_id=TENANT_B)
    profile = _profile(["e1"])

    await projection_a.reconcile(_context(profile, frozenset({"e1"})))

    scope_b = VectorStoreScope(tenant_id=TENANT_B)
    filter_b = MetadataFilter(
        conditions={
            "user_id": USER_ID,
            "index_domain": LTM_INDEX_DOMAIN,
            "collection_name": _collection_name(TENANT_B),
        }
    )
    tenant_b_ids = VectorstoreManager(store_b, scope=scope_b).list_vector_ids_by_metadata(
        scope=scope_b,
        metadata_filter=filter_b,
        limit=100,
    )
    assert "e2" in tenant_b_ids


class _ListingUnsupportedVectorStore(VectorStore):
    def __init__(self, tenant_id: str) -> None:
        self._tenant_id = tenant_id

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str] | None:
        return [record.vector_id for record in records]

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        return []

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        return None

    def count(self, *, scope: VectorStoreScope) -> int:
        return 0


@pytest.mark.asyncio
async def test_ltm_reconcile_fails_when_metadata_listing_unsupported() -> None:
    projection = _ltm_projection(store=_ListingUnsupportedVectorStore(TENANT_A))
    coordinator = UserProfileMemoryLifecycleCoordinator(projections=(projection,))
    profile = _profile(["e1"])

    outcome = await coordinator.reconcile_user(identity=_RECONCILE_IDENTITY, profile=profile)

    assert outcome.disposition is MemoryReconciliationDisposition.FAILED
    failure = outcome.projection_evidence[0].failure
    assert failure is not None
    assert failure.category is MemoryProjectionFailureCategory.PERMANENT


class _TimeoutListingVectorStore(InMemoryVectorStore):
    def list_vector_ids_by_metadata(
        self,
        *,
        scope: VectorStoreScope,
        metadata_filter: MetadataFilter | None = None,
        limit: int = 10_000,
    ) -> list[str]:
        raise TimeoutError("backend timeout")


@pytest.mark.asyncio
async def test_ltm_reconcile_backend_timeout_surfaces_as_failed() -> None:
    projection = _ltm_projection(store=_TimeoutListingVectorStore(TENANT_A))
    coordinator = UserProfileMemoryLifecycleCoordinator(projections=(projection,))
    profile = _profile(["e1"])

    outcome = await coordinator.reconcile_user(identity=_RECONCILE_IDENTITY, profile=profile)

    assert outcome.disposition is MemoryReconciliationDisposition.FAILED
    failure = outcome.projection_evidence[0].failure
    assert failure is not None
    assert failure.category is MemoryProjectionFailureCategory.RETRYABLE
