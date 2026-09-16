# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.memory.memory_vector_errors import MemoryTenantScopeViolationError
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore

pytestmark = pytest.mark.gate

_BOUND_TENANT = "tenant-bound"
_BOUND_NAMESPACE = "ns-bound"
_BOUND_WORKSPACE = "ws-bound"


@dataclass(frozen=True, slots=True)
class _CapturedScope:
    tenant_id: str
    namespace: str | None
    workspace_id: str | None


@dataclass(frozen=True, slots=True)
class _FakeEmbeddingPort:
    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.1, 0.2] for _ in texts]


class _ScopeCapturingVectorstorePort:
    def __init__(self) -> None:
        self.scopes: list[_CapturedScope] = []

    def add_records(
        self,
        records: Sequence[object],
        *,
        scope: _CapturedScope,
    ) -> None:
        self.scopes.append(
            _CapturedScope(
                tenant_id=scope.tenant_id,
                namespace=scope.namespace,
                workspace_id=scope.workspace_id,
            )
        )

    def delete(self, ids: Sequence[str], *, scope: _CapturedScope) -> None:
        self.scopes.append(
            _CapturedScope(
                tenant_id=scope.tenant_id,
                namespace=scope.namespace,
                workspace_id=scope.workspace_id,
            )
        )

    def query(
        self,
        embedding: Sequence[float],
        *,
        scope: _CapturedScope,
        top_k: int,
        metadata_filter: Mapping[str, str | int | float] | None = None,
    ) -> list:
        self.scopes.append(
            _CapturedScope(
                tenant_id=scope.tenant_id,
                namespace=scope.namespace,
                workspace_id=scope.workspace_id,
            )
        )
        return []


def _store(vectorstore: _ScopeCapturingVectorstorePort) -> VectorSessionTurnIndexStore:
    return VectorSessionTurnIndexStore(
        embedding_port=_FakeEmbeddingPort(),
        vectorstore_port=vectorstore,
        tenant_id=_BOUND_TENANT,
        vector_index_namespace=_BOUND_NAMESPACE,
        workspace_id=_BOUND_WORKSPACE,
    )


@pytest.mark.asyncio
async def test_search_turns_rejects_cross_tenant_scope_override() -> None:
    store = _store(_ScopeCapturingVectorstorePort())
    with pytest.raises(MemoryTenantScopeViolationError):
        await store.search_turns(
            query="hello",
            tenant_id="other-tenant",
            session_id="sess-1",
        )


@pytest.mark.asyncio
async def test_upsert_search_tombstone_use_bound_scope() -> None:
    vectorstore = _ScopeCapturingVectorstorePort()
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
