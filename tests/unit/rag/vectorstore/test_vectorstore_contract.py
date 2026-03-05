import pytest
from typing import List, Optional, Sequence
from langchain_core.documents import Document

from intergrax.rag.vectorstore.contracts.vector_store import (
    VectorStore,
    VectorStoreHit,
    MetadataFilter,
)

pytestmark = pytest.mark.unit


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._store = {}
        self._embeddings = {}

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents/embeddings length mismatch")

        if ids and len(ids) != len(documents):
            raise ValueError("ids length mismatch")

        for i, doc in enumerate(documents):
            doc_id = ids[i] if ids else f"id_{len(self._store)}"
            self._store[doc_id] = doc
            self._embeddings[doc_id] = embeddings[i]

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:

        results = []

        for idx, (doc_id, doc) in enumerate(self._store.items()):
            if metadata_filter:
                for k, v in metadata_filter.conditions.items():
                    if doc.metadata.get(k) != v:
                        break
                else:
                    pass
                if any(
                    doc.metadata.get(k) != v
                    for k, v in metadata_filter.conditions.items()
                ):
                    continue

            results.append(
                VectorStoreHit(
                    id=doc_id,
                    content=doc.page_content,
                    metadata=dict(doc.metadata),
                    similarity_score=1.0,
                    rank=idx,
                    embedding=self._embeddings[doc_id]
                    if include_embeddings
                    else None,
                )
            )

        return results[:top_k]

    def delete(self, ids: Sequence[str]) -> None:
        for i in ids:
            self._store.pop(i, None)
            self._embeddings.pop(i, None)

    def count(self) -> int:
        return len(self._store)


def _docs(n: int):
    return [
        Document(page_content=f"text_{i}", metadata={"group": i % 2})
        for i in range(n)
    ]


def _emb(n: int):
    return [[float(i), float(i + 1)] for i in range(n)]


def test_add_and_count():
    store = InMemoryVectorStore()
    store.add_documents(_docs(5), _emb(5))
    assert store.count() == 5


def test_query_top_k():
    store = InMemoryVectorStore()
    store.add_documents(_docs(10), _emb(10))

    hits = store.query(
        query_embedding=[0.0, 1.0],
        top_k=3,
        include_embeddings=False,
    )

    assert len(hits) == 3
    for idx, h in enumerate(hits):
        assert 0.0 <= h.similarity_score <= 1.0
        assert h.rank == idx
        assert h.embedding is None


def test_metadata_filter():
    store = InMemoryVectorStore()
    store.add_documents(_docs(6), _emb(6))

    hits = store.query(
        query_embedding=[0.0, 1.0],
        top_k=10,
        metadata_filter=MetadataFilter(conditions={"group": 1}),
    )

    assert all(h.metadata["group"] == 1 for h in hits)


def test_delete():
    store = InMemoryVectorStore()
    store.add_documents(_docs(4), _emb(4), ids=["a", "b", "c", "d"])
    store.delete(["a", "b"])
    assert store.count() == 2


def test_length_mismatch():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        store.add_documents(_docs(2), _emb(1))