# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore import vectorstore_manager as vectorstore_manager_module
from intergrax.rag.vectorstore.governance.collection_access_policy import (
    CollectionAccessDenied,
    CollectionAccessPolicy,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _record(
    *,
    tenant_id: str = "t1",
    namespace: str | None = None,
    metadata: dict[str, object] | None = None,
) -> VectorStoreRecord:
    document_scope: dict[str, str] = {"tenant_id": tenant_id}
    if namespace is not None:
        document_scope["namespace"] = namespace
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "d1", "root_document_id": "d1"},
            "scope": document_scope,
            "content": "x",
            "metadata": metadata or {},
            "provenance": {"source_kind": "test", "source_id": "s1"},
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=[0.1, 0.2, 0.3],
        vector_id="d1",
    )


class _LegacyProvider:
    def __init__(self) -> None:
        self.documents: list[object] = []

    def add_documents(
        self,
        *,
        documents: list[object],
        embeddings: list[list[float]],
        ids: list[str],
    ) -> list[str]:
        del embeddings
        self.documents.extend(documents)
        return ids


def test_collection_access_policy_denies_query() -> None:
    store = VectorstoreManager(
        InMemoryVectorStore(tenant_id="t1"),
        access_policy=CollectionAccessPolicy(tenant_id="t1", deny_query=True),
    )
    with pytest.raises(CollectionAccessDenied, match="collection_query_denied"):
        store.query(
            query_embedding=[0.1, 0.2, 0.3],
            scope=VectorStoreScope(tenant_id="t1"),
            top_k=1,
        )


def test_collection_access_policy_denies_ingest() -> None:
    store = VectorstoreManager(
        InMemoryVectorStore(tenant_id="t1"),
        access_policy=CollectionAccessPolicy(tenant_id="t1", deny_ingest=True),
    )
    with pytest.raises(CollectionAccessDenied, match="collection_ingest_denied"):
        store.add_records(
            [_record()],
            scope=VectorStoreScope(tenant_id="t1"),
        )


def test_collection_access_policy_workspace_allowlist() -> None:
    store = VectorstoreManager(
        InMemoryVectorStore(tenant_id="t1"),
        access_policy=CollectionAccessPolicy(
            tenant_id="t1",
            allowed_workspace_ids=frozenset({"ws-allowed"}),
        ),
    )
    with pytest.raises(CollectionAccessDenied, match="workspace_not_allowed"):
        store.query(
            query_embedding=[0.1, 0.2, 0.3],
            scope=VectorStoreScope(tenant_id="t1", workspace_id="ws-denied"),
            top_k=1,
            metadata_filter=None,
        )


def test_bound_workspace_write_ignores_metadata_spoof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _LegacyProvider()
    record = _record(
        tenant_id="tenant-a",
        namespace="rag",
        metadata={"workspace_id": "spoofed-workspace"},
    )
    manager = VectorstoreManager(
        provider,
        scope=VectorStoreScope(
            tenant_id="tenant-a",
            namespace="rag",
            workspace_id="workspace-a",
        ),
        access_policy=CollectionAccessPolicy(
            tenant_id="tenant-a",
            allowed_workspace_ids=frozenset({"workspace-a"}),
        ),
    )
    policy_calls: list[tuple[str, str | None, str | None]] = []
    real_enforce = vectorstore_manager_module.enforce_collection_access

    def policy_spy(
        policy: CollectionAccessPolicy | None,
        operation: str,
        *,
        workspace_id: str | None,
        collection_name: str | None,
    ) -> None:
        policy_calls.append((operation, workspace_id, collection_name))
        real_enforce(
            policy,
            operation,
            workspace_id=workspace_id,
            collection_name=collection_name,
        )

    monkeypatch.setattr(
        vectorstore_manager_module,
        "enforce_collection_access",
        policy_spy,
    )

    manager.add_records([record])

    assert policy_calls == [("write", "workspace-a", None)]
    assert len(provider.documents) == 1
    assert provider.documents[0].metadata["workspace_id"] == "workspace-a"
    assert record.document.metadata["workspace_id"] == "spoofed-workspace"


def test_bound_workspace_rejects_conflicting_explicit_scope_before_provider() -> None:
    provider = _LegacyProvider()
    manager = VectorstoreManager(
        provider,
        scope=VectorStoreScope(
            tenant_id="tenant-a",
            namespace="rag",
            workspace_id="workspace-a",
        ),
    )

    with pytest.raises(VectorStoreContractError, match="workspace_id"):
        manager.add_records(
            [_record(tenant_id="tenant-a", namespace="rag")],
            scope=VectorStoreScope(
                tenant_id="tenant-a",
                namespace="rag",
                workspace_id="workspace-b",
            ),
        )

    assert provider.documents == []


def test_bound_tenant_rejects_mismatched_document_before_provider() -> None:
    provider = _LegacyProvider()
    manager = VectorstoreManager(
        provider,
        scope=VectorStoreScope(
            tenant_id="tenant-a",
            namespace="rag",
            workspace_id="workspace-a",
        ),
    )

    with pytest.raises(VectorStoreContractError, match="tenant_id"):
        manager.add_records([_record(tenant_id="tenant-b", namespace="rag")])

    assert provider.documents == []
