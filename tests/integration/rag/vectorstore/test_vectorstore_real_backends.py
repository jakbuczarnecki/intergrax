import uuid
import pytest
from typing import List

from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter


pytestmark = pytest.mark.integration


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _docs(n: int) -> List[Document]:
    return [
        Document(page_content=f"text_{i}", metadata={"group": i % 2})
        for i in range(n)
    ]


def _emb(n: int, dim: int = 4):
    return [[float(i + j) for j in range(dim)] for i in range(n)]


@pytest.fixture(params=["qdrant", "chroma"])
def store(request):
    if request.param == "qdrant":
        return create_qdrant_vector_store(
            collection_name=_unique_name("it_qdrant"),
            tenant_id="tenant_a",
        )

    if request.param == "chroma":
        return create_chroma_vector_store(
            collection_name=_unique_name("it_chroma"),
            tenant_id="tenant_a",
            mode="http",
            http_host="localhost",
            http_port=8000,
        )

    raise RuntimeError("Unknown backend")


def test_full_lifecycle(store):
    docs = _docs(10)
    embs = _emb(10)

    store.add_documents(docs, embs)
    assert store.count() == 10

    hits = store.query(
        query_embedding=embs[0],
        top_k=3,
        include_embeddings=False,
    )

    assert len(hits) == 3

    for idx, h in enumerate(hits):
        assert h.rank == idx
        assert 0.0 <= h.similarity_score <= 1.0
        assert h.embedding is None

    store.delete([hits[0].id])
    assert store.count() == 9


def test_metadata_filter(store):
    docs = _docs(6)
    embs = _emb(6)

    store.add_documents(docs, embs)

    hits = store.query(
        query_embedding=embs[0],
        top_k=10,
        metadata_filter=MetadataFilter(conditions={"group": 1}),
    )

    assert all(h.metadata["group"] == 1 for h in hits)


def test_tenant_isolation():
    name = _unique_name("tenant_test")

    store_a = create_qdrant_vector_store(collection_name=name, tenant_id="A")
    store_b = create_qdrant_vector_store(collection_name=name, tenant_id="B")

    docs = _docs(5)
    embs = _emb(5)

    store_a.add_documents(docs, embs)

    assert store_a.count() == 5
    assert store_b.count() == 0
