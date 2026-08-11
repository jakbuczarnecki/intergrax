# © Artur Czarnecki. All rights reserved.

"""Mandatory pre-ranking membership proof for scoped indexed Ask."""

from __future__ import annotations

import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    MetadataMembershipCondition,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.tools.providers.rag.contracts import RagRetrieveInput
from intergrax.tools.providers.rag.service import _build_metadata_filter

pytestmark = pytest.mark.unit


def _scope() -> VectorStoreScope:
    return VectorStoreScope(tenant_id="tenant-a", namespace="rag", workspace_id="workspace-a")


def _record(
    vector_id: str,
    *,
    source_id: str,
    embedding: list[float],
    content: str,
) -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {
                "document_id": f"doc-{vector_id}",
                "root_document_id": f"doc-{vector_id}",
            },
            "scope": {
                "tenant_id": "tenant-a",
                "namespace": "rag",
                "workspace_id": "workspace-a",
            },
            "content": content,
            "metadata": {},
            "provenance": {
                "source_kind": "file",
                "source_id": source_id,
            },
        }
    )
    return VectorStoreRecord(document=document, embedding=embedding, vector_id=vector_id)


def test_membership_filter_excludes_higher_scoring_source_before_top_k() -> None:
    store = InMemoryVectorStore(tenant_id="tenant-a")
    scope = _scope()
    store.add_records(
        [
            _record("vec-a1", source_id="source-a", embedding=[1.0, 0.0], content="alpha one"),
            _record("vec-a2", source_id="source-a", embedding=[0.99, 0.01], content="alpha two"),
            _record("vec-b1", source_id="source-b", embedding=[0.7, 0.71], content="beta one"),
        ],
        scope=scope,
    )
    query = [1.0, 0.0]
    scoped_filter = MetadataFilter(
        membership=(
            MetadataMembershipCondition(
                field="source_id",
                allowed_values=("source-b",),
            ),
        )
    )
    hits = store.query(
        query,
        scope=scope,
        top_k=1,
        metadata_filter=scoped_filter,
    )
    assert len(hits) == 1
    assert hits[0].document.provenance.source_id == "source-b"

    unscoped = store.query(query, scope=scope, top_k=1)
    assert unscoped[0].document.provenance.source_id == "source-a"


def test_rag_retrieve_input_builds_source_membership_filter() -> None:
    metadata_filter = _build_metadata_filter(
        RagRetrieveInput(
            query="hello",
            allowed_source_ids=("source-b", "source-a"),
        )
    )
    assert metadata_filter is not None
    assert metadata_filter.membership[0].field == "source_id"
    assert metadata_filter.membership[0].allowed_values == ("source-b", "source-a")
