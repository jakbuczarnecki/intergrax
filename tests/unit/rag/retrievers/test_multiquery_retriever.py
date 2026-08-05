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
from intergrax.rag.retrievers.providers.multiquery_retriever import (
    MultiQueryRetriever,
)
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit, RetrieverQuery
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit


pytestmark = pytest.mark.unit


class FakeEmbeddingManager(BaseEmbeddingManager):

    def embed_one(self, text: str) -> List[float]:
        return [1.0, 0.0]
    
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


def _document(document_id: str, parent: str) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": f"doc-{document_id}",
            "metadata": {"parent_chunk_id": parent},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


    
class FakeVectorStoreManager:

    def __init__(self):
        self.calls = 0

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        self.calls += 1

        return [
            VectorStoreHit(
                vector_id=identifier,
                document=_document(identifier, parent),
                similarity_score=score,
                rank=rank,
                embedding=[1.0, 0.0],
            )
            for rank, (identifier, parent, score) in enumerate(
                [
                    ("a", "docA", 0.95),
                    ("b", "docB", 0.90),
                    ("a", "docA", 0.85),
                ]
            )
        ]
    
def test_multiquery_retriever_expands_queries_and_deduplicates():

    vs = FakeVectorStoreManager()
    em = FakeEmbeddingManager()

    retriever = MultiQueryRetriever(
        vector_store=vs,
        embedding_manager=em,
        num_queries=3,
    )

    query = RetrieverQuery(
        query_text="test query expansion",  # >2 words to trigger expansion
        query_embedding=None,
        top_k=2,
        metadata_filter=None,
        include_embeddings=False,
    )

    results = retriever.retrieve(query)

    assert len(results) == 2

    assert isinstance(results, tuple)
    assert all(isinstance(result, RetrievalHit) for result in results)
    ids = {r.vector_id for r in results}

    assert ids.issubset({"a", "b"})

    # verify multiple vectorstore calls
    assert vs.calls >= 2
    assert [result.rank for result in results] == [0, 1]