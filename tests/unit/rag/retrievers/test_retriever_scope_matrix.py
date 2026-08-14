from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.retrievers.contracts.base_retriever import (
    BaseRetriever,
    RetrieverQuery,
)
from intergrax.rag.retrievers.providers.fusion_retriever import FusionRetriever
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.retrievers.providers.hierarchical_retriever import (
    HierarchicalRetriever,
)
from intergrax.rag.retrievers.providers.hybrid_retriever import HybridRetriever
from intergrax.rag.retrievers.providers.mmr_retriever import MMRRetriever
from intergrax.rag.retrievers.providers.multiquery_retriever import MultiQueryRetriever
from intergrax.rag.retrievers.providers.parent_child_retriever import (
    ParentChildRetriever,
)
from intergrax.rag.retrievers.providers.vector_similarity_retriever import (
    VectorSimilarityRetriever,
)
from intergrax.rag.retrievers.registry.retriever_registry import RetrieverRegistry
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreHit,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter

pytestmark = [pytest.mark.unit, pytest.mark.gate]

SCOPE = VectorStoreScope(
    tenant_id="tenant-a",
    namespace="namespace-a",
    workspace_id="workspace-a",
)
FILTER = MetadataFilter(conditions={"session_id": "session-a"})
ROUTING_KEYS = {"tenant_id", "namespace", "workspace_id"}


class _EmbeddingManager:
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0]


def _document(
    document_id: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": SCOPE.tenant_id,
                "namespace": SCOPE.namespace,
                "workspace_id": SCOPE.workspace_id,
            },
            "content": f"content-{document_id}",
            "metadata": metadata or {},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


def _hit(
    document_id: str,
    *,
    metadata: dict[str, Any] | None = None,
    rank: int = 0,
) -> VectorStoreHit:
    return VectorStoreHit(
        vector_id=document_id,
        document=_document(document_id, metadata=metadata),
        similarity_score=0.9,
        rank=rank,
        embedding=[1.0, 0.0],
    )


@dataclass(frozen=True)
class _Call:
    method: str
    scope: VectorStoreScope | None
    metadata_filter: MetadataFilter | None


class _RecordingVectorStore:
    def __init__(
        self,
        responses: list[list[str]],
        *,
        metadata_by_id: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._responses = responses
        self._metadata_by_id = metadata_by_id or {}
        self.calls: list[_Call] = []

    def _record(
        self,
        method: str,
        *,
        scope: VectorStoreScope | None,
        metadata_filter: MetadataFilter | None,
    ) -> list[VectorStoreHit]:
        self.calls.append(_Call(method, scope, metadata_filter))
        response_index = min(len(self.calls) - 1, len(self._responses) - 1)
        return [
            _hit(
                document_id,
                metadata=self._metadata_by_id.get(document_id),
                rank=rank,
            )
            for rank, document_id in enumerate(self._responses[response_index])
        ]

    def query(
        self,
        query_embedding,
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return self._record(
            "query",
            scope=scope,
            metadata_filter=metadata_filter,
        )

    def supports_native_hybrid_search(self) -> bool:
        return False


class _HybridRecordingVectorStore(_RecordingVectorStore):
    def supports_native_hybrid_search(self) -> bool:
        return True

    def query_hybrid(
        self,
        query_embedding,
        query_text: str,
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> list[VectorStoreHit]:
        return self._record(
            "query_hybrid",
            scope=scope,
            metadata_filter=metadata_filter,
        )


def _query() -> RetrieverQuery:
    return RetrieverQuery(
        query_text="alpha beta gamma",
        query_embedding=[1.0, 0.0],
        top_k=2,
        metadata_filter=FILTER,
        scope=SCOPE,
        include_embeddings=True,
    )


def _assert_scope_matrix(
    store: _RecordingVectorStore,
    *,
    minimum_calls: int = 1,
) -> None:
    assert len(store.calls) >= minimum_calls
    for call in store.calls:
        assert call.scope is SCOPE
        assert call.metadata_filter is not None
        assert not ROUTING_KEYS.intersection(call.metadata_filter.conditions)
        assert call.metadata_filter.conditions["session_id"] == "session-a"


def test_every_vector_backed_strategy_forwards_exact_scope_and_filter() -> None:
    embedding = _EmbeddingManager()

    vector_store = _RecordingVectorStore([["vector"]])
    VectorSimilarityRetriever(vector_store, embedding).retrieve(_query())
    _assert_scope_matrix(vector_store)

    mmr_store = _RecordingVectorStore([["mmr-a", "mmr-b"]])
    MMRRetriever(mmr_store, embedding).retrieve(_query())
    _assert_scope_matrix(mmr_store)

    hybrid_store = _HybridRecordingVectorStore([["hybrid"]])
    HybridRetriever(hybrid_store, embedding).retrieve(_query())
    _assert_scope_matrix(hybrid_store)
    assert hybrid_store.calls[0].method == "query_hybrid"

    hybrid_fallback_store = _RecordingVectorStore([["hybrid-fallback"]])
    HybridRetriever(hybrid_fallback_store, embedding).retrieve(_query())
    _assert_scope_matrix(hybrid_fallback_store)
    assert hybrid_fallback_store.calls[0].method == "query"

    multiquery_store = _RecordingVectorStore(
        [["multi-a"], ["multi-b"], ["multi-c"]]
    )
    MultiQueryRetriever(
        multiquery_store,
        embedding,
        num_queries=3,
    ).retrieve(_query())
    _assert_scope_matrix(multiquery_store, minimum_calls=2)

    parent_child_store = _RecordingVectorStore([["parent-child"]])
    ParentChildRetriever(parent_child_store, embedding).retrieve(_query())
    _assert_scope_matrix(parent_child_store)

    hierarchical_chunks = _RecordingVectorStore(
        [["chunk"], ["parent-child"]],
        metadata_by_id={
            "parent-child": {
                ChunkMetadataKey.PARENT_CHUNK_ID: "parent-a",
            }
        },
    )
    hierarchical_toc = _RecordingVectorStore(
        [["toc"]],
        metadata_by_id={
            "toc": {
                ChunkMetadataKey.PARENT_CHUNK_ID: "parent-a",
            }
        },
    )
    HierarchicalRetriever(
        hierarchical_chunks,
        embedding,
        toc_store=hierarchical_toc,
        k_chunks=1,
        k_toc=1,
        max_toc_parents=1,
    ).retrieve(_query())
    _assert_scope_matrix(hierarchical_chunks, minimum_calls=2)
    _assert_scope_matrix(hierarchical_toc)

    graph_store = _RecordingVectorStore(
        [["seed"], ["extra"]],
    )

    class _Graph:
        def node_ids_for_chunks(self, chunk_ids: set[str]) -> set[str]:
            return {"node-a"}

        def neighbors(self, node_id: str, *, max_hops: int):
            return ()

        def find_nodes(self, *, label_contains: str, limit: int):
            return ()

        def chunk_ids_for_nodes(self, node_ids: set[str]) -> set[str]:
            return {"extra"}

    GraphRagRetriever(graph_store, embedding, _Graph()).retrieve(_query())
    _assert_scope_matrix(graph_store, minimum_calls=2)


def test_fusion_delegates_original_query_object_to_every_child() -> None:
    delegated: list[RetrieverQuery] = []

    class _Child(BaseRetriever):
        @classmethod
        def name(cls) -> str:
            return "scope_child"

        def retrieve(self, query: RetrieverQuery):
            delegated.append(query)
            return ()

    class _ChildA(_Child):
        @classmethod
        def name(cls) -> str:
            return "child-a"

    class _ChildB(_Child):
        @classmethod
        def name(cls) -> str:
            return "child-b"

    registry = RetrieverRegistry()
    registry.register(_ChildA())
    registry.register(_ChildB())
    query = _query()

    FusionRetriever(
        registry,
        retrievers=["child-a", "child-b"],
    ).retrieve(query)

    assert delegated == [query, query]
    assert all(received is query for received in delegated)
