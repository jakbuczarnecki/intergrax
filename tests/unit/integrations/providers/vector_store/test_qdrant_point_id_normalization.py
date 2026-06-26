# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Focused tests for Qdrant point id normalization (LKW.1.9)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    QdrantConfig,
    QdrantVectorStore,
    _LOGICAL_ID_METADATA_KEY,
    _normalize_point_id,
)

pytestmark = pytest.mark.unit


@dataclass
class _FakeQdrantPoint:
    id: str | int
    payload: Dict[str, Any]
    vector: List[float]


class _FakeQdrantClient:
    def __init__(self) -> None:
        self._collections: Dict[str, List[_FakeQdrantPoint]] = {}
        self.last_upsert_points: List[Any] = []

    def get_collection(self, collection_name: str) -> dict[str, str]:
        if collection_name not in self._collections:
            raise RuntimeError("collection_not_found")
        return {"name": collection_name}

    def create_collection(self, *, collection_name: str, vectors_config: Any = None, **_: Any) -> None:
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

    store.add_documents(
        [Document(page_content="chunk text")],
        [[0.1, 0.2, 0.3]],
        ids=[logical_id],
    )

    assert len(client.last_upsert_points) == 1
    point = client.last_upsert_points[0]
    assert str(point.id) != logical_id
    uuid.UUID(str(point.id))


def test_upsert_preserves_logical_id_in_metadata() -> None:
    store, client = _store_with_fake_client()
    logical_id = "ingest-lkw-live-smoke-0"

    store.add_documents(
        [Document(page_content="chunk text")],
        [[0.1, 0.2, 0.3]],
        ids=[logical_id],
    )

    payload = client.last_upsert_points[0].payload
    assert payload[_LOGICAL_ID_METADATA_KEY] == logical_id


def test_upsert_valid_uuid_id_remains_valid() -> None:
    store, client = _store_with_fake_client()
    logical_id = "550e8400-e29b-41d4-a716-446655440000"

    store.add_documents(
        [Document(page_content="chunk text")],
        [[0.1, 0.2, 0.3]],
        ids=[logical_id],
    )

    point = client.last_upsert_points[0]
    assert str(point.id) == logical_id
    assert point.payload[_LOGICAL_ID_METADATA_KEY] == logical_id


def test_add_documents_without_ids_generates_valid_uuid() -> None:
    store, client = _store_with_fake_client()

    store.add_documents(
        [Document(page_content="auto id")],
        [[0.1, 0.2, 0.3]],
    )

    point = client.last_upsert_points[0]
    generated_id = str(point.id)
    uuid.UUID(generated_id)
    assert point.payload[_LOGICAL_ID_METADATA_KEY] == generated_id
