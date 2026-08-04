# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from typing import Any, Dict, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
import pytest
from langchain_core.documents import Document

from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStoreHit


pytestmark = pytest.mark.unit


class FakeEmbeddingManager(BaseEmbeddingManager):

    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed_documents(self, documents):
        raise AssertionError("indexing compatibility must use embed_texts")
    
    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        self.received_texts = list(texts)
        return np.array(
            [[index, index + 1, index + 2] for index in range(len(texts))],
            dtype=np.float32,
        )

    def embed_one(
        self,
        text: str,
    ) -> NDArray[np.float32]:
        pass


class ControlledEmbeddingManager(FakeEmbeddingManager):
    def __init__(self, *, row_delta: int = 0, dimension: int = 3) -> None:
        super().__init__()
        self.row_delta = row_delta
        self.dimension = dimension

    def embed_texts(
        self,
        texts: Sequence[str],
    ) -> NDArray[np.float32]:
        self.received_texts = list(texts)
        return np.ones(
            (len(texts) + self.row_delta, self.dimension),
            dtype=np.float32,
        )


class FakeVectorstore(BaseVectorstoreManager):

    def __init__(self):
        self.docs = []
        self.embeddings = None

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.docs.extend(documents)
        self.embeddings = embeddings
    
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


def test_single_index_strategy_inserts_documents():

    docs = [
        Document(page_content="A"),
        Document(page_content="B"),
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
    assert embed_manager.received_texts == ["A", "B"]
    assert vectorstore.docs == docs
    assert np.array_equal(vectorstore.embeddings, [[0, 1, 2], [1, 2, 3]])


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
        Document(page_content="A"),
        Document(page_content="B"),
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
    assert vectorstore.docs == []
    assert embed_manager.received_texts == ["A", "B"]