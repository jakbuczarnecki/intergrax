# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.memory_vector_errors import MemoryTenantScopeViolationError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope

pytestmark = pytest.mark.gate

_BOUND_TENANT = "tenant-bound"
_BOUND_NAMESPACE = "ns-bound"
_BOUND_WORKSPACE = "ws-bound"


class _ScopeCapturingVectorstore:
    def __init__(self) -> None:
        self.scopes: list[VectorStoreScope] = []

    def add_records(self, records, *, scope: VectorStoreScope) -> None:
        self.scopes.append(scope)

    def delete(self, ids, *, scope: VectorStoreScope) -> None:
        self.scopes.append(scope)

    def query(self, embedding, *, scope: VectorStoreScope, top_k: int, metadata_filter) -> list:
        self.scopes.append(scope)
        return []


def _store(vectorstore: _ScopeCapturingVectorstore) -> VectorSessionTurnIndexStore:
    embedding = MagicMock()
    embedding.embed_texts.return_value = [[0.1, 0.2]]
    return VectorSessionTurnIndexStore(
        embedding_manager=embedding,
        vectorstore_manager=vectorstore,
        tenant_id=_BOUND_TENANT,
        vector_index_namespace=_BOUND_NAMESPACE,
        workspace_id=_BOUND_WORKSPACE,
    )


@pytest.mark.asyncio
async def test_search_turns_rejects_cross_tenant_scope_override() -> None:
    store = _store(_ScopeCapturingVectorstore())
    with pytest.raises(MemoryTenantScopeViolationError):
        await store.search_turns(
            query="hello",
            tenant_id="other-tenant",
            session_id="sess-1",
        )


@pytest.mark.asyncio
async def test_upsert_search_tombstone_use_bound_scope() -> None:
    vectorstore = _ScopeCapturingVectorstore()
    store = _store(vectorstore)
    message = ChatMessage(role="user", content="hello", entry_id="entry-1")

    await store.upsert_turn(
        tenant_id=_BOUND_TENANT,
        session_id="sess-1",
        user_id="user-1",
        message=message,
    )
    await store.search_turns(
        query="hello",
        tenant_id=_BOUND_TENANT,
        session_id="sess-1",
    )
    await store.tombstone_turn("entry-1")

    assert len(vectorstore.scopes) == 3
    for scope in vectorstore.scopes:
        assert scope.tenant_id == _BOUND_TENANT
        assert scope.namespace == _BOUND_NAMESPACE
        assert scope.workspace_id == _BOUND_WORKSPACE


@pytest.mark.asyncio
async def test_upsert_rejects_cross_tenant() -> None:
    store = _store(_ScopeCapturingVectorstore())
    message = ChatMessage(role="user", content="hello", entry_id="entry-1")
    with pytest.raises(MemoryTenantScopeViolationError):
        await store.upsert_turn(
            tenant_id="other-tenant",
            session_id="sess-1",
            user_id="user-1",
            message=message,
        )
