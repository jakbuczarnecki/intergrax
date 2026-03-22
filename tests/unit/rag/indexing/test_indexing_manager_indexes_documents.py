# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import pytest
from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.embedding.contracts.embedding_result import EmbeddingResult
from intergrax.rag.indexing.indexing_manager import IndexingManager
from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit


pytestmark = pytest.mark.unit


class FakeEmbeddingManager(BaseEmbeddingManager):

    def embed_documents(self, documents):
        vectors = [[0.1, 0.2, 0.3] for _ in documents]
        return vectors, documents
    
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
        documents: Sequence[Document],
    ) -> EmbeddingResult:
        vectors = [[0.1, 0.2, 0.3] for _ in documents]
        return EmbeddingResult(
            documents=documents,
            embeddings=vectors,
        )


class FakeVectorstore(BaseVectorstoreManager):

    def __init__(self):
        self.docs = []

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.docs.extend(documents)
    
    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> Sequence[VectorStoreHit]:
        pass
    
    def delete(self, ids: Sequence[str]) -> None:
        pass
    
    def count(self) -> int:
        return len(self.docs)


def test_indexing_manager_indexes_documents():

    docs = [Document(page_content="doc")]

    embed = FakeEmbeddingManager()
    store = FakeVectorstore()

    manager = IndexingManager(
        embed_manager=embed,
        vectorstore=store,
        strategy=SingleIndexStrategy(),
    )

    manager.index_documents(docs)

    assert store.count() == 1