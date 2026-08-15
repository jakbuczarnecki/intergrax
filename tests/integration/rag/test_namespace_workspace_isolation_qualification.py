from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import pytest

from intergrax.utils import attribute_access

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.integrations.providers.vector_store.qdrant.rag_store import (
    QdrantConfig,
    QdrantVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import (
    create_default_retriever_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.gate

TENANT = "tenant-scope-8"
SCOPE_A1 = VectorStoreScope(
    tenant_id=TENANT,
    namespace="knowledge-a",
    workspace_id="workspace-1",
)
SCOPE_A2 = VectorStoreScope(
    tenant_id=TENANT,
    namespace="knowledge-a",
    workspace_id="workspace-2",
)
SCOPE_B1 = VectorStoreScope(
    tenant_id=TENANT,
    namespace="knowledge-b",
    workspace_id="workspace-1",
)
SCOPE_B2 = VectorStoreScope(
    tenant_id=TENANT,
    namespace="knowledge-b",
    workspace_id="workspace-2",
)

FIXTURE = (
    (SCOPE_A1, "ALLOWED_A1", 0.70),
    (SCOPE_A2, "FORBIDDEN_A2", 0.99),
    (SCOPE_B1, "FORBIDDEN_B1", 0.98),
    (SCOPE_B2, "FORBIDDEN_B2", 0.97),
)


class _EmbeddingManager:
    def embed_one(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]


def _embedding_for(score: float) -> list[float]:
    return [score, math.sqrt(1.0 - score**2)]


def _record(scope: VectorStoreScope, marker: str, score: float) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": marker,
                "root_document_id": marker,
            },
            "scope": {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            },
            "content": marker,
            "metadata": {"marker": marker},
            "provenance": {"source_kind": "test", "source_id": marker},
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=_embedding_for(score),
        vector_id=marker,
    )


def _inmemory_manager() -> VectorstoreManager:
    manager = VectorstoreManager(InMemoryVectorStore(tenant_id=TENANT))
    for scope, marker, score in FIXTURE:
        manager.add_records([_record(scope, marker, score)], scope=scope)
    return manager


def _assert_exact_scope(scope: VectorStoreScope, document: KnowledgeDocument) -> None:
    assert (
        document.scope.tenant_id,
        document.scope.namespace,
        document.scope.workspace_id,
    ) == (
        scope.tenant_id,
        scope.namespace,
        scope.workspace_id,
    )


def test_inmemory_namespace_workspace_negative_gate_is_adversarial_and_symmetric() -> (
    None
):
    assert all(
        forbidden_score > 0.70
        for _, marker, forbidden_score in FIXTURE
        if marker != "ALLOWED_A1"
    )

    manager = _inmemory_manager()
    for scope, expected_marker, _ in FIXTURE:
        hits = manager.query([1.0, 0.0], scope=scope, top_k=4)
        returned_markers = [hit.document.metadata["marker"] for hit in hits]

        assert returned_markers == [expected_marker]
        assert set(returned_markers).isdisjoint(
            {"FORBIDDEN_A2", "FORBIDDEN_B1", "FORBIDDEN_B2"} - {expected_marker}
        )
        for hit in hits:
            _assert_exact_scope(scope, hit.document)

    assert (
        manager.query(
            [1.0, 0.0],
            scope=SCOPE_A1,
            top_k=4,
            metadata_filter=MetadataFilter(conditions={"marker": "FORBIDDEN_A2"}),
        )
        == []
    )
    with pytest.raises(VectorStoreContractError, match="reserved routing key"):
        MetadataFilter(conditions={"namespace": "knowledge-b"})


def test_bound_manager_resolves_scope_and_fails_closed_on_namespace_or_workspace_escape() -> (
    None
):
    manager = VectorstoreManager(
        InMemoryVectorStore(tenant_id=TENANT),
        scope=SCOPE_A1,
    )
    manager.add_records([_record(SCOPE_A1, "ALLOWED_A1", 0.70)])

    hits = manager.query([1.0, 0.0], top_k=4)
    assert [hit.document.metadata["marker"] for hit in hits] == ["ALLOWED_A1"]
    _assert_exact_scope(SCOPE_A1, hits[0].document)

    for escaped_scope in (SCOPE_A2, SCOPE_B1, SCOPE_B2):
        with pytest.raises(VectorStoreContractError):
            manager.query([1.0, 0.0], scope=escaped_scope, top_k=4)


def test_retrieval_service_exposes_only_requested_scope_in_both_directions() -> None:
    vector_store = _inmemory_manager()
    profile = RagProfile(
        route_mode="off",
        retriever_id="vector_similarity",
        enable_rerank=False,
    )
    retriever_manager = create_default_retriever_manager(
        vector_store=vector_store,
        embedding_manager=_EmbeddingManager(),
        profile=profile,
        discover_entry_points=False,
    )
    service = RetrievalService(retriever_manager=retriever_manager, profile=profile)

    for scope, expected_marker in (
        (SCOPE_A1, "ALLOWED_A1"),
        (SCOPE_B2, "FORBIDDEN_B2"),
    ):
        result = service.retrieve(
            RetrievalRequest(query="scope-qualified retrieval", scope=scope, top_k=4)
        )
        assert result.used is True
        assert [chunk.text for chunk in result.chunks] == [expected_marker]
        assert "FORBIDDEN_A2" not in {chunk.text for chunk in result.chunks}
        assert "FORBIDDEN_B1" not in {chunk.text for chunk in result.chunks}
        assert all(
            chunk.scope
            == {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            }
            for chunk in result.chunks
        )


@dataclass
class _FakeQdrantPoint:
    id: str | int
    payload: dict[str, Any]
    vector: list[float]


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.points: list[_FakeQdrantPoint] = []
        self.query_filters: list[Any] = []

    def get_collection(self, collection_name: str) -> dict[str, str]:
        del collection_name
        if not self.points:
            raise RuntimeError("collection_not_found")
        return {"name": "fake"}

    def create_collection(
        self, *, collection_name: str, vectors_config: Any = None, **_: Any
    ) -> None:
        del collection_name, vectors_config

    def upsert(self, *, collection_name: str, points: Sequence[Any]) -> None:
        del collection_name
        self.points.extend(
            _FakeQdrantPoint(
                id=point.id,
                payload=dict(point.payload or {}),
                vector=list(point.vector),
            )
            for point in points
        )

    @staticmethod
    def _condition_value(condition: Any) -> tuple[str | None, Any]:
        if isinstance(condition, dict):
            match = condition.get("match") or {}
            return condition.get("key"), match.get("value")
        match = attribute_access.optional(condition, "match", None)
        return attribute_access.optional(condition, "key", None), attribute_access.optional(
            match, "value", None
        )

    def query_points(
        self,
        *,
        query: Sequence[float],
        query_filter: Any,
        limit: int,
        **_: Any,
    ) -> Any:
        self.query_filters.append(query_filter)
        rows = list(self.points)
        for condition in attribute_access.optional(query_filter, "must", []) or []:
            key, value = self._condition_value(condition)
            rows = [row for row in rows if row.payload.get(key) == value]

        def score(row: _FakeQdrantPoint) -> float:
            return sum(left * right for left, right in zip(query, row.vector))

        rows.sort(key=score, reverse=True)
        selected = rows[:limit]

        class _Result:
            def __init__(self, selected_rows: Sequence[_FakeQdrantPoint]) -> None:
                self.points = [
                    type(
                        "Point",
                        (),
                        {
                            "id": row.id,
                            "payload": dict(row.payload),
                            "score": score(row),
                            "vector": list(row.vector),
                        },
                    )()
                    for row in selected_rows
                ]

        return _Result(selected)


def test_qdrant_scope_filter_is_exact_and_executes_server_side() -> None:
    pytest.importorskip("qdrant_client")
    store = QdrantVectorStore(QdrantConfig(collection_name="scope-8", tenant_id=TENANT))
    client = _FakeQdrantClient()
    store._client = client  # type: ignore[attr-defined]

    for scope, marker, score in FIXTURE:
        store.add_records([_record(scope, marker, score)], scope=scope)

    hits = store.query([1.0, 0.0], scope=SCOPE_A1, top_k=4)

    assert [hit.vector_id for hit in hits] == ["ALLOWED_A1"]
    _assert_exact_scope(SCOPE_A1, hits[0].document)
    conditions = {
        key: value
        for key, value in (
            _FakeQdrantClient._condition_value(condition)
            for condition in client.query_filters[-1].must
        )
    }
    assert conditions == {
        "tenant_id": TENANT,
        "namespace": "knowledge-a",
        "workspace_id": "workspace-1",
    }
