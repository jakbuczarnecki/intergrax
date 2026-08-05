# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from typing import List, Optional, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.retrievers.providers.vector_similarity_retriever import (
    VectorSimilarityRetriever,
)
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit


pytestmark = pytest.mark.unit


class FakeEmbeddingManager(BaseEmbeddingManager):

    def embed_one(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3]
    
    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        pass

    def embed_documents(
        self,
        documents: Sequence[KnowledgeDocument],
    ) -> EmbeddingResult:
        pass


def _document(document_id: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": f"doc-{document_id}",
            "metadata": {"source": "test"},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


class FakeVectorStoreManager:

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        return [
            VectorStoreHit(
                vector_id=identifier,
                document=_document(identifier),
                similarity_score=score,
                rank=rank,
            )
            for rank, (identifier, score) in enumerate(
                [("a", 0.9), ("b", 0.8), ("c", 0.7)]
            )
        ]
    
def test_vector_similarity_retriever_basic():

    vs = FakeVectorStoreManager()
    em = FakeEmbeddingManager()

    retriever = VectorSimilarityRetriever(
        vector_store=vs,
        embedding_manager=em,
    )

    query = RetrieverQuery(
        query_text="test query",
        query_embedding=None,
        top_k=2,
        metadata_filter=None,
        include_embeddings=False,
    )

    results = retriever.retrieve(query)

    assert len(results) == 2

    assert isinstance(results, tuple)
    assert all(isinstance(result, RetrievalHit) for result in results)
    assert results[0].vector_id == "a"
    assert results[1].vector_id == "b"
    assert results[0].score == 0.9
    assert results[0].rank == 0
    assert results[0].document.scope.tenant_id == "tenant-a"
    assert results[0].embedding is None
    assert results[0].document.metadata["source"] == "test"