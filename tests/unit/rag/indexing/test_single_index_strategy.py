# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.indexing.strategies.dual_index_strategy import DualIndexStrategy
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

    def __init__(self) -> None:
        self.received_documents: tuple[KnowledgeDocument, ...] = ()

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        raise AssertionError("native indexing must use embed_documents")

    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        pass

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        self.received_documents = tuple(documents)
        return EmbeddingResult(
            documents=self.received_documents,
            embeddings=np.array(
                [
                    [index, index + 1, index + 2]
                    for index in range(len(documents))
                ],
                dtype=np.float32,
            ),
        )


class ControlledEmbeddingManager(FakeEmbeddingManager):
    def __init__(self, *, row_delta: int = 0, dimension: int = 3) -> None:
        super().__init__()
        self.row_delta = row_delta
        self.dimension = dimension

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        self.received_documents = tuple(documents)
        return EmbeddingResult(
            documents=self.received_documents,
            embeddings=np.ones(
                (len(documents) + self.row_delta, self.dimension),
                dtype=np.float32,
            ),
        )


class FakeVectorstore(BaseVectorstoreManager):

    def __init__(self):
        self.records = []
        self.embeddings = None
        self.calls = []

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope | None = None,
    ) -> None:
        self.calls.append({"records": records, "scope": scope})
        self.records.extend(records)
        self.embeddings = np.array([record.embedding for record in records])
    
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


def test_single_index_strategy_inserts_documents():

    docs = [
        _native_document("A", 0),
        _native_document("B", 1),
    ]

    embed_manager = FakeEmbeddingManager()
    vectorstore = FakeVectorstore()

    strategy = SingleIndexStrategy()

    strategy.build_index(
        documents=docs,
        embed_manager=embed_manager,
        vectorstore=vectorstore,
    )

    assert vectorstore.count() == 2
    assert len(vectorstore.calls) == 1
    assert vectorstore.calls[0]["scope"] is None
    assert vectorstore.calls[0]["records"] == vectorstore.records
    assert embed_manager.received_documents == tuple(docs)
    assert [record.document.content for record in vectorstore.records] == ["A", "B"]
    assert all(isinstance(record, VectorStoreRecord) for record in vectorstore.records)
    assert vectorstore.records[0].document.scope.tenant_id == "tenant.test"
    assert np.array_equal(vectorstore.embeddings, [[0, 1, 2], [1, 2, 3]])


def test_dual_index_strategy_batches_without_operational_scope():
    docs = [
        _native_document("A", 0),
        _native_document("B", 1),
    ]
    embed_manager = FakeEmbeddingManager()
    vectorstore = FakeVectorstore()
    toc_vectorstore = FakeVectorstore()

    DualIndexStrategy(toc_vectorstore=toc_vectorstore, batch_size=1).build_index(
        documents=docs,
        embed_manager=embed_manager,
        vectorstore=vectorstore,
    )

    assert [call["scope"] for call in vectorstore.calls] == [None, None]
    assert [
        record.document.content
        for call in vectorstore.calls
        for record in call["records"]
    ] == ["A", "B"]
    assert toc_vectorstore.calls == []


@pytest.mark.parametrize(
    ("row_delta", "dimension"),
    [
        (-1, 3),
        (0, 0),
    ],
)
def test_single_index_strategy_rejects_invalid_embeddings_before_vectorstore(
    row_delta: int,
    dimension: int,
) -> None:
    docs = [
        _native_document("A", 0),
        _native_document("B", 1),
    ]
    embed_manager = ControlledEmbeddingManager(
        row_delta=row_delta,
        dimension=dimension,
    )
    vectorstore = FakeVectorstore()

    with pytest.raises(ValueError):
        SingleIndexStrategy().build_index(
            documents=docs,
            embed_manager=embed_manager,
            vectorstore=vectorstore,
        )

    assert vectorstore.count() == 0
    assert vectorstore.records == []
    assert embed_manager.received_documents == tuple(docs)


def _native_document(content: str, index: int) -> KnowledgeDocument:
    document_id = f"document-{index}"
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {"tenant_id": "tenant.test", "namespace": "rag"},
            "content": content,
            "metadata": {"position": index},
            "provenance": {
                "source_kind": "test",
                "source_id": f"source-{index}",
            },
        }
    )