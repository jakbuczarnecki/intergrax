from __future__ import annotations

from typing import Any

import pytest

from intergrax.integrations.providers.vector_store.chroma.rag_store import (
    ChromaConfig,
    ChromaVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit


class _FakeChromaCollection:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.where_calls: list[dict[str, Any] | None] = []
        self.query_calls = 0

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        for vector_id, embedding, metadata, document in zip(
            ids,
            embeddings,
            metadatas,
            documents,
        ):
            self.rows[vector_id] = {
                "embedding": embedding,
                "metadata": metadata,
                "document": document,
            }

    def query(self, **_: Any) -> None:
        self.query_calls += 1
        raise AssertionError("ownership lookup must not use similarity query")

    def get(
        self,
        *,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, list[str]]:
        del include
        self.where_calls.append(where)
        return {
            "ids": [
                vector_id
                for vector_id, row in self.rows.items()
                if self._matches(row["metadata"], where)
            ]
        }

    @classmethod
    def _matches(
        cls,
        metadata: dict[str, Any],
        where: dict[str, Any] | None,
    ) -> bool:
        if not where:
            return True
        if "$and" in where:
            return all(cls._matches(metadata, item) for item in where["$and"])
        return all(
            metadata.get(key) == condition["$eq"] for key, condition in where.items()
        )


class _FakeChromaClient:
    def __init__(self, collection: _FakeChromaCollection) -> None:
        self.collection = collection

    def get_or_create_collection(self, **_: Any) -> _FakeChromaCollection:
        return self.collection


def _scope(workspace_id: str) -> VectorStoreScope:
    return VectorStoreScope(
        tenant_id="tenant-a",
        namespace="namespace-a",
        workspace_id=workspace_id,
    )


def _record(
    vector_id: str, source_id: str, scope: VectorStoreScope
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
            "content": vector_id,
            "metadata": {},
            "provenance": {"source_kind": "file", "source_id": source_id},
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=[1.0, 0.0],
        vector_id=vector_id,
    )


def test_chroma_source_lookup_uses_exact_metadata_get_and_logical_ids() -> None:
    collection = _FakeChromaCollection()
    store = ChromaVectorStore(
        ChromaConfig(collection_name="ownership", tenant_id="tenant-a"),
        client=_FakeChromaClient(collection),
    )
    scope_a = _scope("workspace-a")
    scope_b = _scope("workspace-b")
    source_a = "C:/docs/report.md"

    returned_ids = store.add_records(
        [
            _record("logical-a-2", source_a, scope_a),
            _record("logical-b", "D:/docs/report.md", scope_a),
            _record("logical-a-1", source_a, scope_a),
        ],
        scope=scope_a,
    )
    store.add_records(
        [_record("logical-a-other-scope", source_a, scope_b)],
        scope=scope_b,
    )

    assert returned_ids == [
        "logical-a-2",
        "logical-b",
        "logical-a-1",
    ]
    assert store.list_source_record_ids(
        source_id=source_a,
        scope=scope_a,
    ) == ("logical-a-1", "logical-a-2")
    assert store.list_source_record_ids(
        source_id=source_a,
        scope=scope_b,
    ) == ("logical-a-other-scope",)
    assert collection.query_calls == 0
    assert collection.where_calls
    assert "source_id" in str(collection.where_calls[0])
    assert "workspace-a" in str(collection.where_calls[0])


def test_chroma_runtime_errors_after_open_are_not_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = _FakeChromaCollection()
    store = ChromaVectorStore(
        ChromaConfig(collection_name="runtime", tenant_id="tenant-a"),
        client=_FakeChromaClient(collection),
    )
    scope = _scope("workspace-runtime")
    record = _record("runtime-id", "source://runtime", scope)

    def fail(*_: Any, **__: Any) -> None:
        raise RuntimeError("backend runtime failure")

    monkeypatch.setattr(collection, "upsert", fail)
    with pytest.raises(RuntimeError, match="backend runtime failure"):
        store.add_records([record], scope=scope)

    monkeypatch.setattr(collection, "query", fail)
    with pytest.raises(RuntimeError, match="backend runtime failure"):
        store.query([1.0, 0.0], scope=scope, top_k=1)

    monkeypatch.setattr(collection, "get", fail)
    with pytest.raises(RuntimeError, match="backend runtime failure"):
        store.count(scope=scope)
    with pytest.raises(RuntimeError, match="backend runtime failure"):
        store.list_source_record_ids(source_id="source://runtime", scope=scope)

    monkeypatch.setattr(collection, "delete", fail, raising=False)
    with pytest.raises(RuntimeError, match="backend runtime failure"):
        store.delete(["runtime-id"], scope=scope)
