# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pytest
from intergrax.knowledge.contracts import KnowledgeDocument
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
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
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


class _FakeChromaCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _matches(metadata: dict[str, Any], where: Any) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(
                _FakeChromaCollection._matches(metadata, item)
                for item in where["$and"]
            )
        return all(
            metadata.get(key)
            == (value.get("$eq") if isinstance(value, dict) else value)
            for key, value in where.items()
        )

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
        documents: Sequence[str],
    ) -> None:
        for row_id, embedding, metadata, document in zip(
            ids, embeddings, metadatas, documents
        ):
            self.rows[row_id] = {
                "embedding": list(embedding),
                "metadata": dict(metadata),
                "document": document,
            }

    def query(
        self,
        *,
        n_results: int,
        where: Any = None,
        include: Sequence[str],
        **_: Any,
    ) -> dict[str, Any]:
        selected = [
            (row_id, row)
            for row_id, row in self.rows.items()
            if self._matches(row["metadata"], where)
        ][:n_results]
        result: dict[str, Any] = {
            "ids": [[row_id for row_id, _ in selected]],
            "metadatas": [[row["metadata"] for _, row in selected]],
            "documents": [[row["document"] for _, row in selected]],
            "distances": [[0.0 for _ in selected]],
        }
        if "embeddings" in include:
            result["embeddings"] = [[row["embedding"] for _, row in selected]]
        return result

    def delete(self, *, ids: Sequence[str], where: Any = None) -> None:
        for row_id in list(ids):
            if row_id in self.rows and self._matches(self.rows[row_id]["metadata"], where):
                del self.rows[row_id]

    def get(self, *, where: Any = None, include: Sequence[str]) -> dict[str, Any]:
        return {
            "ids": [
                row_id
                for row_id, row in self.rows.items()
                if self._matches(row["metadata"], where)
            ]
        }


class _FakeChromaClient:
    def __init__(self) -> None:
        self.collection = _FakeChromaCollection()

    def get_or_create_collection(
        self,
        *,
        name: str,
        **_: Any,
    ) -> _FakeChromaCollection:
        del name
        return self.collection


_QDRANT_FAKE_CLIENT = _FakeQdrantClient()
_CHROMA_FAKE_CLIENT = _FakeChromaClient()


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
        client=_CHROMA_FAKE_CLIENT,
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
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "x", "root_document_id": "x"},
            "scope": {"tenant_id": "tenant_B"},
            "content": "x",
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": "x"},
        }
    )
    with pytest.raises(ValueError, match="scope"):
        store.add_records(
            [
                VectorStoreRecord(
                    document=document,
                    embedding=[0.1, 0.2, 0.3],
                    vector_id="x",
                )
            ],
            scope=VectorStoreScope(tenant_id="tenant_A"),
        )
