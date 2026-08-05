# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from typing import List, Optional, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.retrievers.providers.parent_child_retriever import (
    ParentChildRetriever,
)
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit


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
            "content": f"chunk-{document_id}",
            "metadata": {ChunkMetadataKey.PARENT_CHUNK_ID: parent},
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
                document=_document(identifier, parent),
                similarity_score=score,
                rank=rank,
                embedding=[1.0, 0.0],
            )
            for rank, (identifier, parent, score) in enumerate(
                [
                    ("a1", "docA", 0.95),
                    ("a2", "docA", 0.94),
                    ("b1", "docB", 0.93),
                    ("b2", "docB", 0.92),
                ]
            )
        ]
    
def test_parent_child_retriever_groups_by_parent():

    vs = FakeVectorStoreManager()
    em = FakeEmbeddingManager()

    retriever = ParentChildRetriever(
        vector_store=vs,
        embedding_manager=em,
        max_per_parent=1,
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

    parents = {r.parent_vector_id for r in results}

    assert parents == {"docA", "docB"}
    assert [result.rank for result in results] == [0, 1]