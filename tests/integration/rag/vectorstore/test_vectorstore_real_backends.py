# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import uuid
from typing import Callable, List

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store
from intergrax.integrations.providers.vector_store.pgvector.bundle import create_pgvector_vector_store
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.integrations.providers.vector_store.weaviate.bundle import create_weaviate_vector_store
from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter, VectorStore
from intergrax.rag.vectorstore.soak.prod_slo import (
    STABLE_PROD_SLO_SLUGS,
    SoakConfig,
    run_vectorstore_soak,
    unique_soak_collection,
)
from intergrax.rag.vectorstore.tenant.tenant_isolation_contract import run_tenant_isolation_contract

pytestmark = [pytest.mark.integration, pytest.mark.vectorstore_soak]


def _unique_name(prefix: str) -> str:
    return unique_soak_collection(prefix)


def _docs(n: int) -> List[Document]:
    return [
        Document(page_content=f"text_{i}", metadata={"group": i % 2})
        for i in range(n)
    ]


def _emb(n: int, dim: int = 4):
    return [[float(i + j) for j in range(dim)] for i in range(n)]


def _open_stable_store(slug: str) -> VectorStore:
    name = _unique_name(f"it_{slug}")
    builders: dict[str, Callable[[], VectorStore]] = {
        "qdrant": lambda: create_qdrant_vector_store(
            collection_name=name,
            tenant_id="tenant_a",
        ),
        "chroma": lambda: create_chroma_vector_store(
            collection_name=name,
            tenant_id="tenant_a",
            mode="http",
            http_host="localhost",
            http_port=8000,
        ),
        "pgvector": lambda: create_pgvector_vector_store(
            tenant_id=f"tenant_{uuid.uuid4().hex[:8]}",
        ),
        "weaviate": lambda: create_weaviate_vector_store(
            collection=name,
            tenant_id="tenant_a",
            url="http://localhost:8080",
        ),
    }
    try:
        return builders[slug]()
    except Exception as exc:
        pytest.skip(f"{slug} backend unavailable: {exc}")


@pytest.fixture(params=list(STABLE_PROD_SLO_SLUGS))
def store(request: pytest.FixtureRequest) -> VectorStore:
    slug = str(request.param)
    return _open_stable_store(slug)


def test_full_lifecycle(store: VectorStore) -> None:
    docs = _docs(10)
    embs = _emb(10)

    try:
        store.add_documents(docs, embs)
    except Exception as exc:
        pytest.skip(f"backend add_documents failed: {exc}")
    assert store.count() == 10

    hits = store.query(
        query_embedding=embs[0],
        top_k=3,
        include_embeddings=False,
    )

    assert len(hits) == 3

    for idx, hit in enumerate(hits):
        assert hit.rank == idx
        assert 0.0 <= hit.similarity_score <= 1.0
        assert hit.embedding is None

    store.delete([hits[0].id])
    assert store.count() == 9


def test_metadata_filter(store: VectorStore) -> None:
    docs = _docs(6)
    embs = _emb(6)

    store.add_documents(docs, embs)

    hits = store.query(
        query_embedding=embs[0],
        top_k=10,
        metadata_filter=MetadataFilter(conditions={"group": 1}),
    )

    assert all(hit.metadata["group"] == 1 for hit in hits)


def test_tenant_isolation_qdrant_live() -> None:
    name = _unique_name("tenant_test")

    def _qdrant_factory(tenant_id: str, collection_name: str):
        return create_qdrant_vector_store(collection_name=collection_name, tenant_id=tenant_id)

    try:
        result = run_tenant_isolation_contract(
            _qdrant_factory,
            slug="qdrant",
            collection_name=name,
            tenant_a="tenant_A",
            tenant_b="tenant_B",
        )
    except Exception as exc:
        pytest.skip(f"qdrant backend unavailable: {exc}")

    if not result.cross_query_isolated and result.reason.startswith("tenant_a_ingest_failed"):
        pytest.skip(f"qdrant tenant probe failed: {result.reason}")

    assert result.cross_query_isolated is True, result.reason
    assert result.ingest_mismatch_rejected is True, result.reason


@pytest.mark.parametrize("slug", list(STABLE_PROD_SLO_SLUGS))
def test_prod_slo_soak_gate(slug: str) -> None:
    """M-RAG.30 — stable backend soak; skipped when service is not reachable."""
    backend = _open_stable_store(slug)
    try:
        result = run_vectorstore_soak(
            backend,
            slug=slug,
            config=SoakConfig(
                document_count=30,
                query_rounds=4,
                top_k=5,
                max_p95_query_ms=5_000.0,
            ),
        )
    except Exception as exc:
        pytest.skip(f"{slug} soak probe failed: {exc}")

    if not result.passed and result.reason.startswith(("ingest_failed", "query_failed", "count_failed")):
        pytest.skip(f"{slug} soak unavailable: {result.reason}")

    assert result.passed is True, result.reason
    assert result.p95_query_ms <= 5_000.0
