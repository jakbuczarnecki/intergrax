# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.chroma.rag_store import ChromaConfig, ChromaVectorStore
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.integrations.providers.vector_store.lancedb.bundle import create_lancedb_vector_store
from intergrax.integrations.providers.vector_store.pgvector.rag_store import PgVectorRagStore
from intergrax.integrations.providers.vector_store.qdrant.rag_store import QdrantConfig, QdrantVectorStore
from intergrax.integrations.providers.vector_store.typesense.bundle import create_typesense_vector_store
from intergrax.integrations.providers.vector_store.weaviate.rag_store import (
    WeaviateConfig,
    WeaviateVectorStore,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from intergrax.rag.vectorstore.tenant.tenant_isolation_contract import (
    TENANT_ISOLATION_CONTRACT_BACKENDS,
    run_tenant_isolation_contract,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


@dataclass
class _FakeQdrantPoint:
    id: str
    payload: Dict[str, Any]
    vector: List[float]
    score: float = 1.0


class _FakeQdrantClient:
    """Minimal in-process Qdrant client for tenant isolation contract tests."""

    def __init__(self) -> None:
        self._collections: Dict[str, List[_FakeQdrantPoint]] = {}

    def get_collection(self, collection_name: str) -> dict[str, str]:
        if collection_name not in self._collections:
            raise RuntimeError("collection_not_found")
        return {"name": collection_name}

    def create_collection(self, *, collection_name: str, vectors_config: Any = None, **_: Any) -> None:
        self._collections.setdefault(collection_name, [])

    def upsert(self, *, collection_name: str, points: Sequence[Any]) -> None:
        bucket = self._collections.setdefault(collection_name, [])
        for point in points:
            bucket.append(
                _FakeQdrantPoint(
                    id=str(point.id),
                    payload=dict(point.payload or {}),
                    vector=list(point.vector),
                )
            )

    def query_points(
        self,
        *,
        collection_name: str,
        query_filter: Any = None,
        limit: int = 10,
        **_: Any,
    ) -> Any:
        points = list(self._collections.get(collection_name, []))
        must = attribute_access.optional(query_filter, "must", None) or []
        for condition in must:
            key = condition.get("key") if isinstance(condition, dict) else attribute_access.optional(condition, "key", None)
            match = condition.get("match") if isinstance(condition, dict) else attribute_access.optional(condition, "match", None)
            value = match.get("value") if isinstance(match, dict) else attribute_access.optional(match, "value", None)
            if key:
                points = [p for p in points if p.payload.get(key) == value]
        selected = points[:limit]

        class _Result:
            def __init__(self, rows: Sequence[_FakeQdrantPoint]) -> None:
                self.points = [
                    type(
                        "Point",
                        (),
                        {
                            "id": row.id,
                            "payload": dict(row.payload),
                            "score": row.score,
                            "vector": list(row.vector),
                        },
                    )()
                    for row in rows
                ]

        return _Result(selected)


_QDRANT_FAKE_CLIENT = _FakeQdrantClient()


def _factory_inmemory(tenant_id: str, collection_name: str) -> VectorStore:
    del collection_name
    return InMemoryVectorStore(tenant_id=tenant_id)


def _factory_pgvector(tenant_id: str, collection_name: str) -> VectorStore:
    del collection_name
    return PgVectorRagStore(tenant_id=tenant_id, dsn=None)


def _factory_weaviate(tenant_id: str, collection_name: str) -> VectorStore:
    return WeaviateVectorStore(
        WeaviateConfig(collection_name=collection_name, tenant_id=tenant_id),
        client=None,
    )


def _factory_qdrant(tenant_id: str, collection_name: str) -> VectorStore:
    store = QdrantVectorStore(
        QdrantConfig(collection_name=collection_name, tenant_id=tenant_id),
    )
    store._client = _QDRANT_FAKE_CLIENT  # type: ignore[attr-defined]
    return store


def _factory_chroma(tenant_id: str, collection_name: str) -> VectorStore:
    return ChromaVectorStore(
        ChromaConfig(collection_name=collection_name, tenant_id=tenant_id),
    )


def _factory_lancedb(tenant_id: str, collection_name: str) -> VectorStore:
    del collection_name
    return create_lancedb_vector_store(vector_store=InMemoryVectorStore(tenant_id=tenant_id))


def _factory_typesense(tenant_id: str, collection_name: str) -> VectorStore:
    del collection_name
    return create_typesense_vector_store(vector_store=InMemoryVectorStore(tenant_id=tenant_id))


_BACKENDS: dict[str, Any] = {
    "inmemory": _factory_inmemory,
    "pgvector": _factory_pgvector,
    "weaviate": _factory_weaviate,
    "qdrant": _factory_qdrant,
    "chroma": _factory_chroma,
    "lancedb": _factory_lancedb,
    "typesense": _factory_typesense,
}


@pytest.mark.parametrize("slug", list(TENANT_ISOLATION_CONTRACT_BACKENDS))
def test_vectorstore_tenant_isolation_contract(slug: str) -> None:
    result = run_tenant_isolation_contract(
        _BACKENDS[slug],
        slug=slug,
        collection_name=f"iso_{slug}",
    )
    assert result.cross_query_isolated is True, result.reason
    assert result.ingest_mismatch_rejected is True, result.reason
    assert result.reason == "ok"


def test_tenant_isolation_backends_match_plan_register() -> None:
    assert set(TENANT_ISOLATION_CONTRACT_BACKENDS) == set(_BACKENDS.keys())


def test_inmemory_rejects_metadata_tenant_mismatch_on_ingest() -> None:
    store = InMemoryVectorStore(tenant_id="tenant_A")
    with pytest.raises(ValueError, match="tenant_id mismatch"):
        store.add_documents(
            [Document(page_content="x", metadata={"tenant_id": "tenant_B"})],
            [[0.1, 0.2, 0.3]],
        )
