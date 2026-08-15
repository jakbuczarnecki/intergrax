# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import pytest
from langchain_core.documents import Document

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.indexing.contracts.index_strategy import IndexStrategy
from intergrax.rag.indexing.indexing_manager import IndexingManager
from intergrax.rag.indexing.pipeline.indexing_pipeline import IndexingPipeline
from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)


pytestmark = pytest.mark.unit


class FakeEmbeddingManager(BaseEmbeddingManager):

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        pass

    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        pass

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        vectors = [[0.1, 0.2, 0.3] for _ in documents]
        return EmbeddingResult(
            documents=documents,
            embeddings=np.asarray(vectors, dtype=np.float32),
        )


class FakeVectorstore(BaseVectorstoreManager):

    def __init__(self):
        self.records = []

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope | None = None,
    ) -> Sequence[str]:
        self.records.extend(records)
        return [record.vector_id for record in records]
    
    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope | None = None,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        pass
    
    def delete(
        self,
        ids: Sequence[str],
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        pass
    
    def count(self, *, scope: VectorStoreScope | None = None) -> int:
        return len(self.records)


class CapturingStrategy(IndexStrategy):
    def __init__(self) -> None:
        self.received: Sequence[KnowledgeDocument] | None = None

    def build_index(
        self,
        *,
        documents: Sequence[KnowledgeDocument],
        embed_manager: BaseEmbeddingManager,
        vectorstore: BaseVectorstoreManager,
    ) -> Sequence[str]:
        del embed_manager, vectorstore
        self.received = documents
        return []


def test_indexing_manager_indexes_documents():

    docs = [_native_document("doc", 0)]

    embed = FakeEmbeddingManager()
    store = FakeVectorstore()

    manager = IndexingManager(
        embed_manager=embed,
        vectorstore=store,
        strategy=SingleIndexStrategy(),
    )

    persisted_ids = manager.index_documents(docs)

    assert store.count() == 1
    assert list(persisted_ids) == [docs[0].identity.document_id]
    assert isinstance(store.records[0], VectorStoreRecord)
    assert store.records[0].document.content == "doc"


def test_indexing_manager_materializes_generator_and_revalidates_documents() -> None:
    strategy = CapturingStrategy()
    manager = IndexingManager(
        embed_manager=FakeEmbeddingManager(),
        vectorstore=FakeVectorstore(),
        strategy=strategy,
    )
    documents = [_native_document("A", 0), _native_document("B", 1)]

    manager.index_documents(document for document in documents)

    assert strategy.received is not None
    assert tuple(document.content for document in strategy.received) == ("A", "B")
    assert tuple(strategy.received) == tuple(documents)
    assert strategy.received[0] is not documents[0]


def test_indexing_manager_rejects_foreign_document_before_strategy() -> None:
    strategy = CapturingStrategy()
    manager = IndexingManager(
        embed_manager=FakeEmbeddingManager(),
        vectorstore=FakeVectorstore(),
        strategy=strategy,
    )

    with pytest.raises(TypeError, match="KnowledgeDocument"):
        manager.index_documents([Document(page_content="legacy")])

    assert strategy.received is None


def test_indexing_manager_empty_input_does_not_call_strategy() -> None:
    strategy = CapturingStrategy()
    manager = IndexingManager(
        embed_manager=FakeEmbeddingManager(),
        vectorstore=FakeVectorstore(),
        strategy=strategy,
    )

    manager.index_documents(iter(()))

    assert strategy.received is None


def test_indexing_pipeline_passes_native_sequence_without_revalidation() -> None:
    strategy = CapturingStrategy()
    pipeline = IndexingPipeline(strategy=strategy)
    documents = (_native_document("A", 0),)

    pipeline.run(
        documents=documents,
        embed_manager=FakeEmbeddingManager(),
        vectorstore=FakeVectorstore(),
    )

    assert strategy.received is documents


def _native_document(content: str, index: int) -> KnowledgeDocument:
    document_id = f"document-{index}"
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant.test"},
            "content": content,
            "metadata": {"position": index},
            "provenance": {
                "source_kind": "test",
                "source_id": f"source-{index}",
            },
        }
    )