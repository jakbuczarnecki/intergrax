# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from langchain_core.documents import Document
import numpy as np
from numpy.typing import NDArray
import pytest
from typing import List, Optional, Sequence

from intergrax.rag.document_splitters.contracts.chunk_metadata_key import ChunkMetadataKey
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.retrievers.providers.hybrid_retriever import HybridRetriever
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

    def __init__(self, id: str, parent: str, score: float):
        self.id = id
        self.content = f"doc-{id}"
        self.metadata = {
            ChunkMetadataKey.PARENT_CHUNK_ID: parent
        }
        self.similarity_score = score
        self.embedding = [1.0, 0.0]
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
            FakeHit("a", "docA", 0.95),
            FakeHit("b", "docB", 0.90),
            FakeHit("c", "docC", 0.80),
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


def test_hybrid_retriever_basic():

    vs = FakeVectorStoreManager()
    em = FakeEmbeddingManager()

    retriever = HybridRetriever(
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

    ids = {r.id for r in results}

    assert ids.issubset({"a", "b", "c"})

    for r in results:
        assert r.metadata is not None