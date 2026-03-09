# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from langchain_core.documents import Document
import numpy as np
from numpy.typing import NDArray
import pytest
from typing import List, Optional, Sequence

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.retrievers.providers.vector_similarity_retriever import (
    VectorSimilarityRetriever,
)
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit


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
        documents: Sequence[Document],
    ) -> Sequence[Document]:
        pass


class FakeHit:

    def __init__(self, id: str, score: float):
        self.id = id
        self.content = f"doc-{id}"
        self.metadata = {"source": "test"}
        self.similarity_score = score
        self.embedding = None
        self.rank = None


class FakeVectorStoreManager(BaseVectorstoreManager):

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        return [
            FakeHit("a", 0.9),
            FakeHit("b", 0.8),
            FakeHit("c", 0.7),
        ]
    
    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        pass


    def delete(self, ids: Sequence[str]) -> None:
        pass

    
    def count(self) -> int:
        pass


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

    assert results[0].id == "a"
    assert results[1].id == "b"

    assert results[0].metadata["source"] == "test"