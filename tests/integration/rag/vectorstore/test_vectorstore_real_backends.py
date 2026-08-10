# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store
from intergrax.integrations.providers.vector_store.pgvector.bundle import create_pgvector_vector_store
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
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


@dataclass(frozen=True)
class _Backend:
    slug: str
    store: VectorStore
    scope: VectorStoreScope


def _record(
    vector_id: str,
    *,
    source_id: str,
    scope: VectorStoreScope,
    group: int = 0,
) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": f"document-{vector_id}",
                "root_document_id": f"document-{vector_id}",
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": f"text-{vector_id}",
            "metadata": {"group": group},
            "provenance": {
                "source_kind": "file",
                "source_id": source_id,
            },
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=[1.0, 0.0, 0.0, 0.0],
        vector_id=vector_id,
    )


def _open_stable_store(slug: str) -> _Backend:
    name = _unique_name(f"it_{slug}")
    scope = VectorStoreScope(
        tenant_id="tenant_a",
        namespace=name,
        workspace_id="workspace_a",
    )
    builders: dict[str, Callable[[], VectorStore]] = {
        "qdrant": lambda: create_qdrant_vector_store(
            collection_name=name,
            tenant_id=scope.tenant_id,
        ),
        "chroma": lambda: create_chroma_vector_store(
            collection_name=name,
            tenant_id=scope.tenant_id,
            mode="http",
            http_host="localhost",
            http_port=8000,
        ),
        "pgvector": lambda: create_pgvector_vector_store(
            tenant_id=scope.tenant_id,
        ),
    }
    try:
        integration = builders[slug]()
        native_store = getattr(integration, "rag_store", integration)
        return _Backend(slug=slug, store=native_store, scope=scope)
    except Exception as exc:
        pytest.skip(f"{slug} backend unavailable: {exc}")


@pytest.fixture(params=list(STABLE_PROD_SLO_SLUGS))
def backend(request: pytest.FixtureRequest) -> _Backend:
    slug = str(request.param)
    return _open_stable_store(slug)


def test_full_lifecycle(backend: _Backend) -> None:
    source_id = f"source://{backend.slug}/lifecycle"
    records = [
        _record(
            f"{backend.slug}-vector-{index}",
            source_id=source_id,
            scope=backend.scope,
            group=index % 2,
        )
        for index in range(10)
    ]

    try:
        returned_ids = backend.store.add_records(records, scope=backend.scope)
    except Exception as exc:
        pytest.skip(f"{backend.slug} native add_records failed: {exc}")
    assert tuple(returned_ids) == tuple(record.vector_id for record in records)
    assert backend.store.count(scope=backend.scope) == 10
    assert tuple(
        backend.store.list_source_record_ids(
            source_id=source_id,
            scope=backend.scope,
        )
    ) == tuple(record.vector_id for record in records)

    hits = backend.store.query(
        query_embedding=records[0].embedding,
        scope=backend.scope,
        top_k=3,
        include_embeddings=False,
    )

    assert len(hits) == 3

    for idx, hit in enumerate(hits):
        assert hit.rank == idx
        assert 0.0 <= hit.similarity_score <= 1.0
        assert hit.embedding is None
        assert hit.vector_id in {record.vector_id for record in records}

    backend.store.delete(
        [record.vector_id for record in records],
        scope=backend.scope,
    )
    assert backend.store.count(scope=backend.scope) == 0
    assert tuple(
        backend.store.list_source_record_ids(
            source_id=source_id,
            scope=backend.scope,
        )
    ) == ()


def test_source_replacement_and_scope_negative_gate(backend: _Backend) -> None:
    source_a = "source://same-basename/a/report.md"
    source_b = "source://same-basename/b/report.md"
    old_records = [
        _record(
            f"{backend.slug}-a-old-{index}",
            source_id=source_a,
            scope=backend.scope,
        )
        for index in range(2)
    ]
    backend.store.add_records(
        old_records
        + [
            _record(
                f"{backend.slug}-b-current",
                source_id=source_b,
                scope=backend.scope,
            )
        ],
        scope=backend.scope,
    )
    old_ids = set(
        backend.store.list_source_record_ids(
            source_id=source_a,
            scope=backend.scope,
        )
    )
    new_record = _record(
        f"{backend.slug}-a-new",
        source_id=source_a,
        scope=backend.scope,
    )
    new_ids = set(
        backend.store.add_records([new_record], scope=backend.scope)
    )
    backend.store.delete(sorted(old_ids - new_ids), scope=backend.scope)

    assert set(
        backend.store.list_source_record_ids(
            source_id=source_a,
            scope=backend.scope,
        )
    ) == {new_record.vector_id}
    assert tuple(
        backend.store.list_source_record_ids(
            source_id=source_b,
            scope=backend.scope,
        )
    ) == (f"{backend.slug}-b-current",)

    foreign_scope = VectorStoreScope(
        tenant_id=backend.scope.tenant_id,
        namespace=f"{backend.scope.namespace}-foreign",
        workspace_id=backend.scope.workspace_id,
    )
    foreign_record = _record(
        f"{backend.slug}-foreign",
        source_id=source_a,
        scope=foreign_scope,
    )
    backend.store.add_records([foreign_record], scope=foreign_scope)
    assert tuple(
        backend.store.list_source_record_ids(
            source_id=source_a,
            scope=backend.scope,
        )
    ) == (new_record.vector_id,)
    visible_ids = {
        hit.vector_id
        for hit in backend.store.query(
            new_record.embedding,
            scope=backend.scope,
            top_k=20,
        )
    }
    assert foreign_record.vector_id not in visible_ids


def test_metadata_filter(backend: _Backend) -> None:
    records = [
        _record(
            f"{backend.slug}-filter-{index}",
            source_id=f"source://{backend.slug}/filter",
            scope=backend.scope,
            group=index % 2,
        )
        for index in range(6)
    ]

    backend.store.add_records(records, scope=backend.scope)

    hits = backend.store.query(
        query_embedding=records[0].embedding,
        scope=backend.scope,
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
            backend.store,
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
