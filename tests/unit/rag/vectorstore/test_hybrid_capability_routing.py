from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QdrantVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    QdrantConfig,
    QdrantVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.hybrid_retriever import HybridRetriever
from intergrax.rag.vectorstore.contracts.hybrid_search import (
    HybridSearchCapable,
    NativeHybridSearchProvider,
    provider_supports_native_hybrid_search,
    resolve_native_hybrid_search_provider,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


SCOPE = VectorStoreScope(
    tenant_id="tenant-a",
    namespace="namespace-a",
    workspace_id="workspace-a",
)
FILTER = MetadataFilter(conditions={"session_id": "session-a"})


class _EmbeddingManager(BaseEmbeddingManager):
    def embed_one(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_texts(self, texts: Sequence[str]):
        raise NotImplementedError

    def embed_documents(self, documents: Sequence[KnowledgeDocument]):
        raise NotImplementedError


def _document(document_id: str, *, content: str) -> KnowledgeDocument:
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
            "content": content,
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


def _hit(document_id: str, *, content: str, score: float, rank: int) -> VectorStoreHit:
    return VectorStoreHit(
        vector_id=document_id,
        document=_document(document_id, content=content),
        similarity_score=score,
        rank=rank,
    )


@dataclass
class _Call:
    method: str
    scope: VectorStoreScope | None
    metadata_filter: MetadataFilter | None


class _NativeHybridProvider:
    def __init__(self) -> None:
        self.calls: list[_Call] = []

    def supports_native_hybrid_search(self) -> bool:
        return True

    def query_hybrid(
        self,
        query_embedding,
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> list[VectorStoreHit]:
        self.calls.append(_Call("query_hybrid", scope, metadata_filter))
        return [_hit("native", content="native hybrid", score=0.99, rank=0)]

    def query(
        self,
        query_embedding,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        self.calls.append(_Call("query", scope, metadata_filter))
        return [_hit("dense", content="dense only", score=0.5, rank=0)]


class _DenseOnlyProvider:
    def __init__(self, hits: list[VectorStoreHit]) -> None:
        self._hits = hits
        self.calls: list[_Call] = []

    def query(
        self,
        query_embedding,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        self.calls.append(_Call("query", scope, metadata_filter))
        return list(self._hits[:top_k])


def _retriever_query() -> RetrieverQuery:
    return RetrieverQuery(
        query_text="alpha unique marker",
        query_embedding=[1.0, 0.0],
        top_k=2,
        metadata_filter=FILTER,
        scope=SCOPE,
        include_embeddings=False,
    )


def test_native_hybrid_provider_is_called_through_manager() -> None:
    provider = _NativeHybridProvider()
    manager = VectorstoreManager(provider, scope=SCOPE)
    embedding = _EmbeddingManager()

    hits = HybridRetriever(manager, embedding).retrieve(_retriever_query())

    assert provider.calls[0].method == "query_hybrid"
    assert provider.calls[0].scope == SCOPE
    assert provider.calls[0].metadata_filter is not None
    assert provider.calls[0].metadata_filter.conditions["session_id"] == "session-a"
    assert hits[0].vector_id == "native"
    assert manager.supports_native_hybrid_search() is True


def test_dense_only_provider_uses_generic_hybrid_rerank() -> None:
    provider = _DenseOnlyProvider(
        [
            _hit(
                "older-proof",
                content="older near identical proof source",
                score=0.95,
                rank=0,
            ),
            _hit(
                "fresh-source",
                content="alpha unique marker persisted payload",
                score=0.70,
                rank=1,
            ),
        ]
    )
    manager = VectorstoreManager(provider, scope=SCOPE)
    embedding = _EmbeddingManager()

    hits = HybridRetriever(manager, embedding).retrieve(_retriever_query())

    assert provider.calls[0].method == "query"
    assert provider.calls[0].scope == SCOPE
    assert provider.calls[0].metadata_filter is not None
    assert provider.calls[0].metadata_filter.conditions["session_id"] == "session-a"
    assert manager.supports_native_hybrid_search() is False
    assert hits[0].vector_id == "fresh-source"


def test_qdrant_integration_wrapper_preserves_sparse_native_capability() -> None:
    inner = QdrantVectorStore(
        QdrantConfig(
            collection_name="proof",
            tenant_id="tenant-a",
            enable_sparse_vectors=True,
        )
    )
    integration = QdrantVectorStoreIntegration.from_store(
        store_config=object(),
        inner=inner,
    )

    assert provider_supports_native_hybrid_search(integration) is True
    assert isinstance(integration, HybridSearchCapable)


def test_qdrant_integration_without_sparse_does_not_claim_native_hybrid() -> None:
    inner = QdrantVectorStore(
        QdrantConfig(
            collection_name="proof",
            tenant_id="tenant-a",
            enable_sparse_vectors=False,
        )
    )
    integration = QdrantVectorStoreIntegration.from_store(
        store_config=object(),
        inner=inner,
    )

    assert provider_supports_native_hybrid_search(integration) is False
    with pytest.raises(IntegrationConfigurationError):
        integration.query_hybrid(
            [1.0, 0.0],
            "alpha",
            scope=SCOPE,
            top_k=5,
        )


def test_manager_query_hybrid_rejects_dense_only_provider() -> None:
    provider = _DenseOnlyProvider([])
    manager = VectorstoreManager(provider, scope=SCOPE)

    with pytest.raises(VectorStoreContractError, match="native hybrid search"):
        manager.query_hybrid(
            [1.0, 0.0],
            "alpha",
            scope=SCOPE,
            top_k=5,
        )


class _DuckTypedHybridWithoutCapability(_DenseOnlyProvider):
    def query_hybrid(
        self,
        query_embedding,
        query_text: str,
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        include_embeddings: bool = False,
        alpha: float = 0.5,
    ) -> list[VectorStoreHit]:
        return []


def test_manager_requires_complete_native_hybrid_provider_contract() -> None:
    provider = _DuckTypedHybridWithoutCapability([])
    manager = VectorstoreManager(provider, scope=SCOPE)

    assert resolve_native_hybrid_search_provider(provider) is None
    assert provider_supports_native_hybrid_search(provider) is False

    with pytest.raises(VectorStoreContractError, match="native hybrid search"):
        manager.query_hybrid(
            [1.0, 0.0],
            "alpha",
            scope=SCOPE,
            top_k=5,
        )


def test_native_provider_satisfies_complete_typed_contract() -> None:
    provider = _NativeHybridProvider()

    assert isinstance(provider, NativeHybridSearchProvider)
    assert resolve_native_hybrid_search_provider(provider) is provider


def test_cold_start_generic_hybrid_ordering_is_stable_without_lexical_cache() -> None:
    persisted_hits = [
        _hit(
            "older-proof",
            content="older near identical proof source",
            score=0.95,
            rank=0,
        ),
        _hit(
            "fresh-source",
            content="alpha unique marker persisted payload",
            score=0.70,
            rank=1,
        ),
    ]
    provider = _DenseOnlyProvider(persisted_hits)
    manager = VectorstoreManager(provider, scope=SCOPE)
    embedding = _EmbeddingManager()
    retriever = HybridRetriever(manager, embedding, prefetch_factor=1)

    before_restart = retriever.retrieve(_retriever_query())
    after_restart = retriever.retrieve(_retriever_query())

    assert [hit.vector_id for hit in before_restart] == [
        hit.vector_id for hit in after_restart
    ]
    assert before_restart[0].vector_id == "fresh-source"


def test_inprocess_empty_coordinator_hides_generation_managed_records() -> None:
    from intergrax.distributed.source_operation import (
        InProcessSourceOperationCoordinator,
        SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
    )

    coordinator = InProcessSourceOperationCoordinator()
    versioned_document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": "fresh-source",
                "root_document_id": "fresh-source",
            },
            "scope": {
                "tenant_id": SCOPE.tenant_id,
                "namespace": SCOPE.namespace,
                "workspace_id": SCOPE.workspace_id,
            },
            "content": "alpha unique marker persisted payload",
            "metadata": {
                SOURCE_PUBLICATION_GENERATION_METADATA_KEY: "1:cold-start-token",
            },
            "provenance": {
                "source_kind": "test",
                "source_id": "/data/user_docs/fresh.txt",
            },
        }
    )
    provider = _DenseOnlyProvider(
        [
            VectorStoreHit(
                vector_id="fresh-source",
                document=versioned_document,
                similarity_score=0.7,
                rank=0,
            )
        ]
    )
    manager = VectorstoreManager(provider, scope=SCOPE)
    manager.set_source_operation_coordinator(coordinator)

    hits = manager.query([1.0, 0.0], scope=SCOPE, top_k=5)

    assert hits == []
