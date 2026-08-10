# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Focused tests for Qdrant point id normalization (LKW.1.9)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    QdrantConfig,
    QdrantVectorStore,
    _LOGICAL_ID_METADATA_KEY,
    _normalize_point_id,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreRecord, VectorStoreScope

pytestmark = pytest.mark.unit


@dataclass
class _FakeQdrantPoint:
    id: str | int
    payload: Dict[str, Any]
    vector: List[float]


class _FakeQdrantClient:
    def __init__(self) -> None:
        self._collections: Dict[str, List[_FakeQdrantPoint]] = {}
        self._collection_dims: Dict[str, int] = {}
        self.last_upsert_points: List[Any] = []
        self.deleted_collections: list[str] = []

    def get_collection(self, collection_name: str) -> Any:
        if collection_name not in self._collection_dims:
            raise RuntimeError("collection_not_found")
        size = self._collection_dims[collection_name]
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=size))
            )
        )

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)
        self._collection_dims.pop(collection_name, None)
        self._collections.pop(collection_name, None)

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: Any = None,
        **_: Any,
    ) -> None:
        if hasattr(vectors_config, "size"):
            dim = int(vectors_config.size)
        elif isinstance(vectors_config, dict):
            dim = int(next(iter(vectors_config.values())).size)
        else:
            dim = 3
        self._collection_dims[collection_name] = dim
        self._collections.setdefault(collection_name, [])

    def upsert(self, *, collection_name: str, points: Sequence[Any]) -> None:
        self.last_upsert_points = list(points)
        bucket = self._collections.setdefault(collection_name, [])
        for point in points:
            bucket.append(
                _FakeQdrantPoint(
                    id=point.id,
                    payload=dict(point.payload or {}),
                    vector=list(point.vector),
                )
            )


def _store_with_fake_client() -> tuple[QdrantVectorStore, _FakeQdrantClient]:
    store = QdrantVectorStore(QdrantConfig(collection_name="coll", tenant_id="t1"))
    client = _FakeQdrantClient()
    store._client = client  # type: ignore[attr-defined]
    store._dim = 3  # type: ignore[attr-defined]
    return store, client


def _record(vector_id: str) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": vector_id, "root_document_id": vector_id},
            "scope": {"tenant_id": "t1"},
            "content": "chunk text",
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": vector_id},
        }
    )
    return VectorStoreRecord(document=document, embedding=[0.1, 0.2, 0.3], vector_id=vector_id)


def test_normalize_point_id_invalid_string_is_deterministic_uuid() -> None:
    raw = "ingest-lkw-live-smoke-0"
    first = _normalize_point_id(raw)
    second = _normalize_point_id(raw)
    uuid.UUID(str(first))
    assert first == second
    assert str(first) != raw


def test_normalize_point_id_valid_uuid_unchanged() -> None:
    raw = "550e8400-e29b-41d4-a716-446655440000"
    assert _normalize_point_id(raw) == raw


def test_normalize_point_id_unsigned_integer_string() -> None:
    assert _normalize_point_id("42") == 42


def test_upsert_normalizes_invalid_string_id() -> None:
    store, client = _store_with_fake_client()
    logical_id = "ingest-lkw-live-smoke-0"

    store.add_records([_record(logical_id)], scope=VectorStoreScope(tenant_id="t1"))

    assert len(client.last_upsert_points) == 1
    point = client.last_upsert_points[0]
    assert str(point.id) != logical_id
    uuid.UUID(str(point.id))


def test_upsert_preserves_logical_id_in_metadata() -> None:
    store, client = _store_with_fake_client()
    logical_id = "ingest-lkw-live-smoke-0"

    store.add_records([_record(logical_id)], scope=VectorStoreScope(tenant_id="t1"))

    payload = client.last_upsert_points[0].payload
    assert payload[_LOGICAL_ID_METADATA_KEY] == logical_id


def test_upsert_valid_uuid_id_remains_valid() -> None:
    store, client = _store_with_fake_client()
    logical_id = "550e8400-e29b-41d4-a716-446655440000"

    store.add_records([_record(logical_id)], scope=VectorStoreScope(tenant_id="t1"))

    point = client.last_upsert_points[0]
    assert str(point.id) == logical_id
    assert point.payload[_LOGICAL_ID_METADATA_KEY] == logical_id


def test_add_records_recreates_collection_on_embedding_dimension_mismatch() -> None:
    store, client = _store_with_fake_client()
    collection_name = store.collection_name
    client._collection_dims[collection_name] = 384
    client._collections[collection_name] = []
    store._dim = 768  # type: ignore[attr-defined]

    record = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-768", "root_document_id": "doc-768"},
            "scope": {"tenant_id": "t1"},
            "content": "dim mismatch",
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": "doc-768"},
        }
    )
    vector = VectorStoreRecord(
        document=record,
        embedding=[0.1] * 768,
        vector_id="doc-768",
    )

    store.add_records([vector], scope=VectorStoreScope(tenant_id="t1"))

    assert collection_name in client.deleted_collections
    assert client._collection_dims[collection_name] == 768
    assert len(client.last_upsert_points) == 1


def test_add_records_requires_explicit_vector_id() -> None:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {"tenant_id": "t1"},
            "content": "auto id",
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": "doc-1"},
        }
    )
    with pytest.raises(ValueError, match="vector_id"):
        VectorStoreRecord(document=document, embedding=[0.1, 0.2, 0.3], vector_id="")
