from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from types import SimpleNamespace

import pytest

from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    QdrantConfig,
    QdrantVectorStore,
    _normalize_point_id,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


def _scope(
    tenant_id: str = "tenant-a",
    namespace: str | None = "rag",
    workspace_id: str | None = "workspace-a",
) -> VectorStoreScope:
    return VectorStoreScope(
        tenant_id=tenant_id,
        namespace=namespace,
        workspace_id=workspace_id,
    )


def _record(
    vector_id: str,
    *,
    source_id: str,
    scope: VectorStoreScope,
    document_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> VectorStoreRecord:
    document_id = document_id or f"document-{vector_id}"
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": f"content-{vector_id}",
            "metadata": metadata or {},
            "provenance": {
                "source_kind": "file",
                "source_id": source_id,
            },
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=[1.0, 0.0],
        vector_id=vector_id,
    )


def test_inmemory_returns_complete_persisted_ids_with_exact_source_scope() -> None:
    source_a = "C:/docs/a/report.md"
    source_b = "D:/docs/b/report.md"
    scope_a = _scope()
    scope_b = _scope(workspace_id="workspace-b")
    scope_other_namespace = _scope(namespace="other-rag")

    manager = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    manager.add_records(
        [
            _record("persisted-a-2", source_id=source_a, scope=scope_a),
            _record("persisted-a-1", source_id=source_a, scope=scope_a),
            _record(
                "persisted-b",
                source_id=source_b,
                scope=scope_a,
                document_id="document-report-b",
            ),
        ],
        scope=scope_a,
    )

    # A record in the other workspace must be ingested through its own scope.
    manager.add_records(
        [_record("persisted-a-other-workspace", source_id=source_a, scope=scope_b)],
        scope=scope_b,
    )
    manager.add_records(
        [_record("persisted-a-other-namespace", source_id=source_a, scope=scope_other_namespace)],
        scope=scope_other_namespace,
    )

    assert manager.list_source_record_ids(source_id=source_a, scope=scope_a) == (
        "persisted-a-1",
        "persisted-a-2",
    )
    assert manager.list_source_record_ids(source_id=source_b, scope=scope_a) == (
        "persisted-b",
    )
    assert (
        manager.list_source_record_ids(source_id=source_a, scope=scope_b)
        == ("persisted-a-other-workspace",)
    )
    assert manager.list_source_record_ids(
        source_id=source_a,
        scope=scope_other_namespace,
    ) == ("persisted-a-other-namespace",)
    assert manager.list_source_record_ids(source_id="missing", scope=scope_a) == ()


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda: InMemoryVectorStore(tenant_id="tenant-a"),
    ],
    ids=["inmemory"],
)
def test_add_ownership_delete_uses_one_logical_id_domain(store_factory: Any) -> None:
    source_id = "C:/docs/lifecycle.md"
    other_source_id = "D:/docs/other.md"
    scope = _scope()
    manager = VectorstoreManager(store_factory(), scope=scope)
    records = [
        _record("logical/lifecycle-1", source_id=source_id, scope=scope),
        _record("logical/lifecycle-2", source_id=source_id, scope=scope),
    ]

    returned = set(manager.add_records(records, scope=scope))
    manager.add_records(
        [_record("logical/other", source_id=other_source_id, scope=scope)],
        scope=scope,
    )
    ownership = set(
        manager.list_source_record_ids(source_id=source_id, scope=scope)
    )

    assert returned == ownership == {
        "logical/lifecycle-1",
        "logical/lifecycle-2",
    }
    lifecycle_ids = {
        "logical/lifecycle-1",
        "logical/lifecycle-2",
    }
    manager.delete(sorted(lifecycle_ids), scope=scope)
    assert manager.list_source_record_ids(source_id=source_id, scope=scope) == ()
    assert manager.list_source_record_ids(
        source_id=other_source_id,
        scope=scope,
    ) == ("logical/other",)


def test_pgvector_without_dsn_fails_closed_instead_of_using_memory() -> None:
    from intergrax.integrations.providers.vector_store.pgvector.rag_store import (
        PgVectorRagStore,
    )

    with pytest.raises(IntegrationConfigurationError, match="DSN"):
        PgVectorRagStore(tenant_id="tenant-a")


def test_source_lookup_is_tenant_isolated_and_bound_scope_cannot_escape() -> None:
    source_id = "C:/docs/report.md"
    scope_a = _scope()
    manager_a = VectorstoreManager(
        InMemoryVectorStore(tenant_id="tenant-a"),
        scope=scope_a,
    )
    manager_a.add_records(
        [_record("tenant-a-vector", source_id=source_id, scope=scope_a)]
    )

    assert manager_a.list_source_record_ids(source_id=source_id) == (
        "tenant-a-vector",
    )
    with pytest.raises(VectorStoreContractError):
        manager_a.list_source_record_ids(
            source_id=source_id,
            scope=_scope(tenant_id="tenant-b"),
        )
    with pytest.raises(VectorStoreContractError):
        manager_a.list_source_record_ids(
            source_id=source_id,
            scope=_scope(workspace_id="workspace-b"),
        )
    with pytest.raises(ValueError, match="source_id"):
        manager_a.list_source_record_ids(source_id=" ")

    manager_b = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-b"))
    scope_b = _scope(tenant_id="tenant-b")
    manager_b.add_records(
        [_record("tenant-b-vector", source_id=source_id, scope=scope_b)],
        scope=scope_b,
    )
    assert manager_b.list_source_record_ids(source_id=source_id, scope=scope_b) == (
        "tenant-b-vector",
    )


def test_source_lookup_reports_unsupported_optional_provider_explicitly() -> None:
    class _UnsupportedProvider:
        _tenant_id = "tenant-a"

    manager = VectorstoreManager(_UnsupportedProvider())  # type: ignore[arg-type]

    with pytest.raises(
        RuntimeError,
        match="vectorstore_source_record_lookup_not_supported",
    ):
        manager.list_source_record_ids(source_id="source", scope=_scope())


def test_source_id_is_system_owned_and_cannot_be_spoofed_by_user_metadata() -> None:
    with pytest.raises(ValueError):
        _record(
            "spoofed",
            source_id="canonical-source",
            scope=_scope(),
            metadata={"source_id": "user-spoof"},
        )


@dataclass
class _FakeQdrantPoint:
    id: str | int
    payload: dict[str, Any]
    vector: list[float]


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.points: list[_FakeQdrantPoint] = []
        self.scroll_calls: list[dict[str, Any]] = []
        self.query_calls = 0
        self.allow_query = False
        self.fail_scroll = False
        self.missing_collection = False

    def get_collection(self, collection_name: str) -> dict[str, str]:
        del collection_name
        if self.missing_collection:
            raise RuntimeError("collection not found")
        return {"name": "fake"}

    def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        del collection_name
        self.points.extend(
            _FakeQdrantPoint(
                id=point.id,
                payload=dict(point.payload or {}),
                vector=list(point.vector),
            )
            for point in points
        )

    def query_points(self, *, query_filter: Any, limit: int, **_: Any) -> Any:
        self.query_calls += 1
        if not self.allow_query:
            raise AssertionError("source ownership must not use similarity query")
        selected = [
            point for point in self.points if self._matches(point, query_filter)
        ]
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id=point.id,
                    payload=point.payload,
                    vector=point.vector,
                    score=1.0,
                )
                for point in selected[:limit]
            ]
        )

    @staticmethod
    def _matches(point: _FakeQdrantPoint, qfilter: Any) -> bool:
        for condition in getattr(qfilter, "must", []) or []:
            has_id = getattr(condition, "has_id", None)
            if has_id is not None and point.id not in has_id:
                return False
            field = getattr(condition, "key", None)
            match = getattr(condition, "match", None)
            if field is not None and match is not None:
                if point.payload.get(field) != getattr(match, "value", None):
                    return False
            is_null = getattr(condition, "is_null", None)
            if is_null is not None:
                if is_null.key in point.payload:
                    return False
        return True

    def delete(self, *, points_selector: Any, **_: Any) -> None:
        qfilter = getattr(points_selector, "filter", None)
        point_ids = getattr(points_selector, "points", None)
        self.points = [
            point
            for point in self.points
            if (
                not self._matches(point, qfilter)
                if qfilter is not None
                else point.id not in (point_ids or [])
            )
        ]

    def scroll(self, *, scroll_filter: Any, offset: Any, **_: Any) -> tuple[list[Any], Any]:
        if self.fail_scroll:
            raise RuntimeError("qdrant backend unavailable")
        self.scroll_calls.append({"filter": scroll_filter})
        start = int(offset or 0)
        selected = [
            point for point in self.points if self._matches(point, scroll_filter)
        ]
        page = selected[start : start + 1]
        next_offset = start + 1 if start + 1 < len(selected) else None
        return page, next_offset


def _qdrant_store() -> tuple[QdrantVectorStore, _FakeQdrantClient]:
    store = QdrantVectorStore(
        QdrantConfig(collection_name="ownership", tenant_id="tenant-a")
    )
    client = _FakeQdrantClient()
    store._client = client  # type: ignore[attr-defined]
    return store, client


def test_qdrant_source_lookup_returns_empty_when_collection_missing() -> None:
    store, client = _qdrant_store()
    client.missing_collection = True

    assert (
        store.list_source_record_ids(
            source_id="C:/docs/a/report.md",
            scope=_scope(),
        )
        == []
    )
    assert client.scroll_calls == []


def test_qdrant_source_lookup_scrolls_complete_native_ids_with_scope_filter() -> None:
    store, client = _qdrant_store()
    scope_a = _scope()
    scope_b = _scope(workspace_id="workspace-b")
    source_a = "C:/docs/a/report.md"

    returned_ids = store.add_records(
        [
            _record("qdrant/a", source_id=source_a, scope=scope_a),
            _record(
                "qdrant/b",
                source_id="D:/docs/b/report.md",
                scope=scope_a,
            ),
            _record("qdrant/a-2", source_id=source_a, scope=scope_a),
        ],
        scope=scope_a,
    )
    store.add_records(
        [_record("qdrant/other-workspace", source_id=source_a, scope=scope_b)],
        scope=scope_b,
    )

    actual_ids = store.list_source_record_ids(source_id=source_a, scope=scope_a)

    assert returned_ids == ["qdrant/a", "qdrant/b", "qdrant/a-2"]
    assert actual_ids == ["qdrant/a", "qdrant/a-2"]
    assert client.query_calls == 0
    assert len(client.scroll_calls) > 1
    conditions = client.scroll_calls[0]["filter"].must
    assert any(
        getattr(condition, "key", None) == "tenant_id"
        and getattr(getattr(condition, "match", None), "value", None) == "tenant-a"
        for condition in conditions
    )
    assert any(
        getattr(condition, "key", None) == "namespace"
        and getattr(getattr(condition, "match", None), "value", None) == "rag"
        for condition in conditions
    )
    assert any(
        getattr(condition, "key", None) == "workspace_id"
        and getattr(getattr(condition, "match", None), "value", None) == "workspace-a"
        for condition in conditions
    )
    assert any(
        getattr(condition, "key", None) == "source_id"
        and getattr(getattr(condition, "match", None), "value", None) == source_a
        for condition in conditions
    )


def test_qdrant_logical_ids_are_queryable_and_delete_maps_with_scope() -> None:
    store, client = _qdrant_store()
    scope_a = _scope()
    scope_b = _scope(workspace_id="workspace-b")
    source_id = "C:/docs/a/report.md"

    store.add_records(
        [_record("logical/a", source_id=source_id, scope=scope_a)],
        scope=scope_a,
    )
    store.add_records(
        [_record("logical/b", source_id=source_id, scope=scope_b)],
        scope=scope_b,
    )
    client.allow_query = True

    hits = store.query(
        [1.0, 0.0],
        scope=scope_a,
        top_k=5,
    )
    assert [hit.vector_id for hit in hits] == ["logical/a"]
    assert str(_normalize_point_id("logical/a")) != "logical/a"

    store.delete(["logical/a"], scope=scope_a)

    assert not any(
        point.id == _normalize_point_id("logical/a") for point in client.points
    )
    assert any(
        point.id == _normalize_point_id("logical/b")
        and point.payload["workspace_id"] == "workspace-b"
        for point in client.points
    )
    assert store.list_source_record_ids(source_id=source_id, scope=scope_a) == []
    assert store.list_source_record_ids(source_id=source_id, scope=scope_b) == [
        "logical/b"
    ]


def test_qdrant_source_lookup_fails_closed_without_logical_id() -> None:
    store, client = _qdrant_store()
    scope = _scope()
    source_id = "C:/docs/a/report.md"
    client.points.append(
        _FakeQdrantPoint(
            id=_normalize_point_id("legacy/a"),
            payload={
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
                "source_id": source_id,
            },
            vector=[1.0, 0.0],
        )
    )

    with pytest.raises(VectorStoreContractError, match="logical vector ID"):
        store.list_source_record_ids(source_id=source_id, scope=scope)


def test_qdrant_source_lookup_returns_empty_for_missing_source_and_propagates_failure() -> None:
    store, client = _qdrant_store()
    scope = _scope()

    assert store.list_source_record_ids(source_id="missing", scope=scope) == []
    with pytest.raises(ValueError, match="source_id"):
        store.list_source_record_ids(source_id="", scope=scope)

    client.fail_scroll = True
    with pytest.raises(RuntimeError, match="backend unavailable"):
        store.list_source_record_ids(source_id="source", scope=scope)


class _FakePgVectorCursor:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def __enter__(self) -> _FakePgVectorCursor:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, statement: str, params: tuple[Any, ...]) -> None:
        self.executed.append((statement, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _FakePgVectorConnection:
    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.cursor_instance = _FakePgVectorCursor(rows)
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _FakePgVectorCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


def _uninitialized_pgvector_store(
    connection: _FakePgVectorConnection,
) -> Any:
    from intergrax.integrations.providers.vector_store.pgvector.rag_store import (
        PgVectorRagStore,
    )

    store = object.__new__(PgVectorRagStore)
    store._tenant_id = "tenant-a"
    store._dimension = 2
    store._connection = connection
    return store


def test_pgvector_uses_server_side_scope_and_cosine_query() -> None:
    connection = _FakePgVectorConnection()
    store = _uninitialized_pgvector_store(connection)
    scope = _scope()

    assert store.query(
        [1.0, 0.0],
        scope=scope,
        top_k=3,
        metadata_filter=None,
    ) == []

    statement, params = connection.cursor_instance.executed[0]
    assert "<=>" in statement
    assert "ORDER BY embedding <=>" in statement
    assert "tenant_id = %s" in statement
    assert "namespace IS NOT DISTINCT FROM %s" in statement
    assert "workspace_id IS NOT DISTINCT FROM %s" in statement
    assert params[1:4] == ("tenant-a", "rag", "workspace-a")


def test_pgvector_source_ownership_and_delete_are_scoped_sql_operations() -> None:
    connection = _FakePgVectorConnection(rows=[("logical-1",), ("logical-2",)])
    store = _uninitialized_pgvector_store(connection)
    scope = _scope()

    assert store.list_source_record_ids(source_id="source-a", scope=scope) == [
        "logical-1",
        "logical-2",
    ]
    ownership_statement, ownership_params = connection.cursor_instance.executed[0]
    assert "SELECT logical_id" in ownership_statement
    assert "source_id = %s" in ownership_statement
    assert "payload->>" not in ownership_statement
    assert ownership_params == ("tenant-a", "rag", "workspace-a", "source-a")

    store.delete(["logical-1"], scope=scope)
    delete_statement, delete_params = connection.cursor_instance.executed[1]
    assert "logical_id = ANY(%s)" in delete_statement
    assert "tenant_id = %s" in delete_statement
    assert "namespace IS NOT DISTINCT FROM %s" in delete_statement
    assert "workspace_id IS NOT DISTINCT FROM %s" in delete_statement
    assert delete_params == (["logical-1"], "tenant-a", "rag", "workspace-a")
    assert connection.commits == 1


def test_pgvector_rejects_incompatible_embedding_dimension() -> None:
    connection = _FakePgVectorConnection()
    store = _uninitialized_pgvector_store(connection)
    scope = _scope()
    two_dimensional = _record("two-dimensional", source_id="source-a", scope=scope)
    invalid = VectorStoreRecord(
        document=two_dimensional.document,
        embedding=[1.0, 0.0, 0.0],
        vector_id="three-dimensional",
    )

    with pytest.raises(IntegrationConfigurationError, match="dimension"):
        store._validate_record_dimensions([invalid])
