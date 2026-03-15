# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from langchain_core.documents import Document
from numpy.typing import NDArray
import pytest
import numpy as np
from typing import List, Optional, Sequence

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.retrievers.providers.mmr_retriever import MMRRetriever
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
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
        documents: Sequence[Document],
    ) -> EmbeddingResult:
        pass


class FakeHit:

    def __init__(self, id: str, emb: List[float], score: float):
        self.id = id
        self.content = f"doc-{id}"
        self.metadata = {}
        self.embedding = emb
        self.similarity_score = score
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
            FakeHit("a", [1.0, 0.0], 0.95),
            FakeHit("b", [0.9, 0.1], 0.94),
            FakeHit("c", [0.0, 1.0], 0.80),
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


def test_mmr_retriever_diversification():

    vs = FakeVectorStoreManager()
    em = FakeEmbeddingManager()

    retriever = MMRRetriever(
        vector_store=vs,
        embedding_manager=em,
    )

    query = RetrieverQuery(
        query_text="test query",
        query_embedding=None,
        top_k=2,
        metadata_filter=None,
        include_embeddings=True,
    )

    results = retriever.retrieve(query)

    assert len(results) == 2

    ids = {r.id for r in results}

    assert "a" in ids
    assert len(ids) == 2