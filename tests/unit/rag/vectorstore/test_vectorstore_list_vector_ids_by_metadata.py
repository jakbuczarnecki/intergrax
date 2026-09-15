# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.gate


def _record(
    *,
    vector_id: str,
    tenant_id: str,
    user_id: str,
) -> VectorStoreRecord:
    import numpy as np

    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": vector_id, "root_document_id": vector_id},
            "scope": {"tenant_id": tenant_id},
            "content": f"text-{vector_id}",
            "metadata": {"user_id": user_id, "index_domain": "ltm"},
            "provenance": {
                "source_kind": "test",
                "source_id": vector_id,
                "source_parent_id": user_id,
            },
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        vector_id=vector_id,
    )


def test_list_vector_ids_by_metadata_respects_scope_tenant_boundary() -> None:
    store_a = InMemoryVectorStore("tenant-a")
    store_b = InMemoryVectorStore("tenant-b")
    scope_a = VectorStoreScope(tenant_id="tenant-a")
    scope_b = VectorStoreScope(tenant_id="tenant-b")
    manager_a = VectorstoreManager(store_a, scope=scope_a)
    manager_b = VectorstoreManager(store_b, scope=scope_b)
    manager_a.add_records([_record(vector_id="e1", tenant_id="tenant-a", user_id="u1")], scope=scope_a)
    manager_b.add_records([_record(vector_id="e2", tenant_id="tenant-b", user_id="u1")], scope=scope_b)

    ids_a = manager_a.list_vector_ids_by_metadata(
        scope=scope_a,
        metadata_filter=MetadataFilter(conditions={"user_id": "u1"}),
        limit=100,
    )

    assert ids_a == ("e1",)
    with pytest.raises(VectorStoreContractError):
        manager_a.list_vector_ids_by_metadata(
            scope=VectorStoreScope(tenant_id="tenant-b"),
            metadata_filter=MetadataFilter(conditions={"user_id": "u1"}),
            limit=100,
        )
