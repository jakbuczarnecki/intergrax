from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreHit,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


def _document(
    document_id: str = "doc-1",
    *,
    tenant_id: str = "tenant-a",
    namespace: str | None = "namespace-a",
    workspace_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": document_id,
                "root_document_id": document_id,
            },
            "scope": {
                "tenant_id": tenant_id,
                "namespace": namespace,
                "workspace_id": workspace_id,
            },
            "content": "native content",
            "metadata": metadata or {},
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


def _scope(
    tenant_id: str = "tenant-a",
    namespace: str | None = "namespace-a",
    workspace_id: str | None = None,
) -> VectorStoreScope:
    return VectorStoreScope(
        tenant_id=tenant_id,
        namespace=namespace,
        workspace_id=workspace_id,
    )


def test_record_revalidates_and_defensively_copies_document_and_vector() -> None:
    document = _document(metadata={"nested": {"value": 1}})
    vector = np.array([1.0, 2.0], dtype=np.float64)
    record = VectorStoreRecord(document=document, embedding=vector, vector_id="external-1")

    assert record.document is not document
    assert record.document.metadata == {"nested": {"value": 1}}
    assert record.embedding.dtype == np.float32
    assert record.embedding.flags.writeable is False
    vector[0] = 99
    assert record.embedding[0] == 1
    with pytest.raises(ValueError):
        record.embedding[0] = 5


@pytest.mark.parametrize(
    "embedding",
    [
        np.array(1.0),
        np.ones((2, 2)),
        np.array([], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_record_rejects_invalid_embedding(embedding: np.ndarray) -> None:
    with pytest.raises(VectorStoreContractError):
        VectorStoreRecord(
            document=_document(),
            embedding=embedding,
            vector_id="doc-1",
        )


def test_record_rejects_foreign_document_and_empty_id() -> None:
    with pytest.raises(TypeError):
        VectorStoreRecord(document=object(), embedding=[1.0], vector_id="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VectorStoreRecord(document=_document(), embedding=[1.0], vector_id=" ")


def test_scope_is_tenant_required_and_immutable() -> None:
    with pytest.raises(ValueError):
        VectorStoreScope(tenant_id=" ")
    scope = _scope(workspace_id="workspace-a")
    with pytest.raises(FrozenInstanceError):
        scope.tenant_id = "tenant-b"  # type: ignore[misc]
    assert scope.namespace == "namespace-a"
    assert scope.workspace_id == "workspace-a"


def test_scope_from_document_and_match_include_workspace() -> None:
    document = _document(workspace_id="workspace-a")
    scope = VectorStoreScope.from_document(document)

    assert scope.workspace_id == "workspace-a"
    assert scope.matches_document(document)
    assert not scope.matches_document(_document(workspace_id="workspace-b"))


def test_metadata_filter_copies_nested_json_and_rejects_routing_keys() -> None:
    conditions = {"nested": {"items": [1, 2]}}
    metadata_filter = MetadataFilter(conditions=conditions)
    conditions["nested"]["items"].append(3)

    assert metadata_filter.conditions["nested"] == {"items": [1, 2]}
    with pytest.raises(ValueError):
        MetadataFilter(conditions={"tenant_id": "spoofed"})
    with pytest.raises(ValueError):
        MetadataFilter(conditions={"items": (1, 2)})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        metadata_filter.conditions["new"] = "value"  # type: ignore[index]


def test_hit_is_native_and_validates_score_rank_and_embedding() -> None:
    hit = VectorStoreHit(
        vector_id="external-1",
        document=_document(),
        similarity_score=1.0,
        rank=0,
        embedding=[1.0, 0.0],
    )
    assert hit.document.identity.document_id == "doc-1"
    assert hit.content == "native content"
    assert hit.id == "external-1"
    assert hit.embedding is not None
    assert hit.embedding.dtype == np.float32
    assert hit.embedding.flags.writeable is False
    with pytest.raises(ValueError):
        VectorStoreHit(
            vector_id="x",
            document=_document(),
            similarity_score=1.1,
            rank=0,
        )
    with pytest.raises(ValueError):
        VectorStoreHit(
            vector_id="x",
            document=_document(),
            similarity_score=1.0,
            rank=True,  # type: ignore[arg-type]
        )


def test_manager_maps_records_and_returns_native_hits() -> None:
    manager = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    scope = _scope()
    manager.add_records(
        [
            VectorStoreRecord(
                document=_document(),
                embedding=[1.0, 0.0],
                vector_id="external-1",
            )
        ],
        scope=scope,
    )

    hits = manager.query(
        [1.0, 0.0],
        scope=scope,
        top_k=1,
        include_embeddings=True,
    )
    assert len(hits) == 1
    assert isinstance(hits[0], VectorStoreHit)
    assert hits[0].document.scope.tenant_id == "tenant-a"
    assert hits[0].vector_id == "external-1"
    assert hits[0].embedding is not None


def test_manager_derives_namespace_for_tenant_bound_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = InMemoryVectorStore(tenant_id="tenant-a")
    received_scopes: list[VectorStoreScope] = []
    real_add_records = provider.add_records

    def capture_add_records(
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        received_scopes.append(scope)
        return real_add_records(records, scope=scope)

    monkeypatch.setattr(provider, "add_records", capture_add_records)
    manager = VectorstoreManager(provider)
    record = VectorStoreRecord(
        document=_document(namespace="rag"),
        embedding=[1.0, 0.0],
        vector_id="namespace-record",
    )

    manager.add_records([record])

    assert received_scopes == [VectorStoreScope(tenant_id="tenant-a", namespace="rag")]
    assert received_scopes[0].namespace == "rag"
    assert record.document.scope.namespace == "rag"
    hits = manager.query(
        [1.0, 0.0],
        scope=VectorStoreScope(tenant_id="tenant-a", namespace="rag"),
        top_k=1,
    )
    assert [hit.vector_id for hit in hits] == ["namespace-record"]
    assert hits[0].document.scope.namespace == "rag"


def test_manager_preserves_bound_workspace_and_rejects_metadata_spoof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = InMemoryVectorStore(tenant_id="tenant-a")
    received_scopes: list[VectorStoreScope] = []
    real_add_records = provider.add_records

    def capture_add_records(
        records: Sequence[VectorStoreRecord],
        *,
        scope: VectorStoreScope,
    ) -> Sequence[str]:
        received_scopes.append(scope)
        return real_add_records(records, scope=scope)

    monkeypatch.setattr(provider, "add_records", capture_add_records)
    bound_scope = VectorStoreScope(
        tenant_id="tenant-a",
        namespace="rag",
        workspace_id="workspace-a",
    )
    manager = VectorstoreManager(provider, scope=bound_scope)
    with pytest.raises(ValueError):
        _document(
            namespace="rag",
            workspace_id="workspace-a",
            metadata={"workspace_id": "workspace-b"},
        )

    record = VectorStoreRecord(
        document=_document(
            namespace="rag",
            workspace_id="workspace-a",
        ),
        embedding=[1.0, 0.0],
        vector_id="workspace-record",
    )
    original_metadata = dict(record.document.metadata)

    manager.add_records([record])

    hits = manager.query([1.0, 0.0], scope=bound_scope, top_k=1)
    assert received_scopes == [bound_scope]
    assert hits[0].document.scope.workspace_id == "workspace-a"
    assert record.document.metadata == original_metadata


def test_manager_rejects_mixed_tenant_before_provider() -> None:
    manager = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    records = [
        VectorStoreRecord(
            document=_document(document_id="a", tenant_id="tenant-a"),
            embedding=[1.0],
            vector_id="a",
        ),
        VectorStoreRecord(
            document=_document(document_id="b", tenant_id="tenant-b"),
            embedding=[1.0],
            vector_id="b",
        ),
    ]
    with pytest.raises(ValueError):
        manager.add_records(records, scope=VectorStoreScope(tenant_id="tenant-a"))
    assert manager.count(scope=VectorStoreScope(tenant_id="tenant-a")) == 0


def test_manager_rejects_mixed_workspaces_before_provider() -> None:
    manager = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    records = [
        VectorStoreRecord(
            document=_document(document_id="a", workspace_id="workspace-a"),
            embedding=[1.0],
            vector_id="a",
        ),
        VectorStoreRecord(
            document=_document(document_id="b", workspace_id="workspace-b"),
            embedding=[1.0],
            vector_id="b",
        ),
    ]

    with pytest.raises(ValueError, match="workspace"):
        manager.add_records(records)


def test_same_document_id_stays_distinct_across_workspace_scopes() -> None:
    manager_a = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    manager_b = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    scope_a = _scope(workspace_id="workspace-a")
    scope_b = _scope(workspace_id="workspace-b")

    manager_a.add_records(
        [
            VectorStoreRecord(
                document=_document(workspace_id="workspace-a"),
                embedding=[1.0],
                vector_id="same-vector-id",
            )
        ],
        scope=scope_a,
    )
    manager_b.add_records(
        [
            VectorStoreRecord(
                document=_document(workspace_id="workspace-b"),
                embedding=[1.0],
                vector_id="same-vector-id",
            )
        ],
        scope=scope_b,
    )

    hit_a = manager_a.query([1.0], scope=scope_a, top_k=1)[0]
    hit_b = manager_b.query([1.0], scope=scope_b, top_k=1)[0]
    assert hit_a.document.identity.document_id == hit_b.document.identity.document_id
    assert hit_a.document.scope.workspace_id == "workspace-a"
    assert hit_b.document.scope.workspace_id == "workspace-b"


def test_tenant_namespace_and_delete_count_are_isolated() -> None:
    store_a = InMemoryVectorStore(tenant_id="tenant-a")
    store_b = InMemoryVectorStore(tenant_id="tenant-b")
    manager_a = VectorstoreManager(store_a)
    manager_b = VectorstoreManager(store_b)
    scope_a = _scope(namespace=None)
    scope_b = _scope(tenant_id="tenant-b", namespace=None)

    manager_a.add_records(
        [
            VectorStoreRecord(
                document=_document(namespace=None),
                embedding=[1.0],
                vector_id="same-id",
            )
        ],
        scope=scope_a,
    )
    manager_b.add_records(
        [
            VectorStoreRecord(
                document=_document(tenant_id="tenant-b", namespace=None),
                embedding=[1.0],
                vector_id="same-id",
            )
        ],
        scope=scope_b,
    )
    assert len(manager_a.query([1.0], scope=scope_a, top_k=1)) == 1
    assert len(manager_b.query([1.0], scope=scope_b, top_k=1)) == 1
    manager_a.delete(["same-id"], scope=scope_a)
    assert manager_a.count(scope=scope_a) == 0
    assert manager_b.count(scope=scope_b) == 1


def test_namespace_query_isolation() -> None:
    manager = VectorstoreManager(InMemoryVectorStore(tenant_id="tenant-a"))
    namespace_a = _scope(namespace="namespace-a")
    namespace_b = _scope(namespace="namespace-b")
    manager.add_records(
        [
            VectorStoreRecord(
                document=_document(document_id="a", namespace="namespace-a"),
                embedding=[1.0],
                vector_id="a",
            )
        ],
        scope=namespace_a,
    )
    manager.add_records(
        [
            VectorStoreRecord(
                document=_document(document_id="b", namespace="namespace-b"),
                embedding=[1.0],
                vector_id="b",
            )
        ],
        scope=namespace_b,
    )
    assert [hit.vector_id for hit in manager.query([1.0], scope=namespace_a, top_k=5)] == ["a"]
    assert [hit.vector_id for hit in manager.query([1.0], scope=namespace_b, top_k=5)] == ["b"]


def test_delete_count_fail_closed_without_tenant_bound_provider() -> None:
    class UnboundProvider:
        def delete(self, ids: object) -> None:
            raise AssertionError("must fail closed")

        def count(self) -> int:
            raise AssertionError("must fail closed")

    manager = VectorstoreManager(
        UnboundProvider(),  # type: ignore[arg-type]
        scope=VectorStoreScope(tenant_id="tenant-a"),
    )
    with pytest.raises(ValueError):
        manager.delete(["id"], scope=VectorStoreScope(tenant_id="tenant-a"))
    with pytest.raises(ValueError):
        manager.count(scope=VectorStoreScope(tenant_id="tenant-a"))
