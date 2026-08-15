import pytest
from typing import List, Optional, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreHit,
    MetadataFilter,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore

pytestmark = pytest.mark.unit


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._store = {}
        self._embeddings = {}

    def add_records(
        self,
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        for record in records:
            if not scope.matches_document(record.document):
                raise ValueError("record scope mismatch")
            self._store[record.vector_id] = record.document
            self._embeddings[record.vector_id] = record.embedding
        return [record.vector_id for record in records]

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        scope: VectorStoreScope,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> List[VectorStoreHit]:

        results = []

        for idx, (doc_id, doc) in enumerate(self._store.items()):
            if not scope.matches_document(doc):
                continue
            if metadata_filter:
                if any(
                    (
                        doc.scope.tenant_id
                        if k == "tenant_id"
                        else doc.scope.namespace
                        if k == "namespace"
                        else doc.metadata.get(k)
                    )
                    != v
                    for k, v in metadata_filter.conditions.items()
                ):
                    continue

            results.append(
                VectorStoreHit(
                    vector_id=doc_id,
                    document=doc,
                    similarity_score=1.0,
                    rank=idx,
                    embedding=self._embeddings[doc_id]
                    if include_embeddings
                    else None,
                )
            )

        return results[:top_k]

    def delete(self, ids: Sequence[str], *, scope: VectorStoreScope) -> None:
        for i in ids:
            if i in self._store and scope.matches_document(self._store[i]):
                self._store.pop(i, None)
                self._embeddings.pop(i, None)

    def count(self, *, scope: VectorStoreScope) -> int:
        return sum(scope.matches_document(doc) for doc in self._store.values())


def _docs(n: int) -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": f"doc-{i}",
                    "root_document_id": f"doc-{i}",
                },
                "scope": {"tenant_id": "tenant-a"},
                "content": f"text_{i}",
                "metadata": {"group": i % 2},
                "provenance": {"source_kind": "test", "source_id": f"doc-{i}"},
            }
        )
        for i in range(n)
    ]


def _emb(n: int):
    return [[float(i), float(i + 1)] for i in range(n)]


def _records(n: int) -> list[VectorStoreRecord]:
    return [
        VectorStoreRecord(document=document, embedding=embedding, vector_id=document.identity.document_id)
        for document, embedding in zip(_docs(n), _emb(n))
    ]


_SCOPE = VectorStoreScope(tenant_id="tenant-a")


def test_add_and_count():
    store = InMemoryVectorStore()
    store.add_records(_records(5), scope=_SCOPE)
    assert store.count(scope=_SCOPE) == 5


def test_query_top_k():
    store = InMemoryVectorStore()
    store.add_records(_records(10), scope=_SCOPE)

    hits = store.query(
        query_embedding=[0.0, 1.0],
        scope=_SCOPE,
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
    store.add_records(_records(6), scope=_SCOPE)

    hits = store.query(
        query_embedding=[0.0, 1.0],
        scope=_SCOPE,
        top_k=10,
        metadata_filter=MetadataFilter(conditions={"group": 1}),
    )

    assert all(h.metadata["group"] == 1 for h in hits)


def test_delete():
    store = InMemoryVectorStore()
    records = _records(4)
    records = [VectorStoreRecord(document=record.document, embedding=record.embedding, vector_id=letter) for record, letter in zip(records, ["a", "b", "c", "d"])]
    store.add_records(records, scope=_SCOPE)
    store.delete(["a", "b"], scope=_SCOPE)
    assert store.count(scope=_SCOPE) == 2


def test_length_mismatch():
    store = InMemoryVectorStore()
    with pytest.raises(ValueError):
        document = _docs(2)[0]
        store.add_records(
            [
                VectorStoreRecord(
                    document=document,
                    embedding=[1.0],
                    vector_id="wrong-scope",
                )
            ],
            scope=VectorStoreScope(tenant_id="tenant-b"),
        )