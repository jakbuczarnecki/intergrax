# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.governance.collection_access_policy import (
    CollectionAccessDenied,
    CollectionAccessPolicy,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _record() -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "d1", "root_document_id": "d1"},
            "scope": {"tenant_id": "t1"},
            "content": "x",
            "metadata": {},
            "provenance": {"source_kind": "test", "source_id": "s1"},
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=[0.1, 0.2, 0.3],
        vector_id="d1",
    )


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
