# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)

pytestmark = pytest.mark.unit


def test_inmemory_query_hybrid_prefers_keyword_match() -> None:
    store = InMemoryVectorStore(tenant_id="t1")
    docs = [
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {"document_id": "a", "root_document_id": "a"},
                "scope": {"tenant_id": "t1"},
                "content": "Intergrax harness RAG pipeline",
                "metadata": {},
                "provenance": {"source_kind": "test", "source_id": "a"},
            }
        ),
        KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {"document_id": "b", "root_document_id": "b"},
                "scope": {"tenant_id": "t1"},
                "content": "unrelated cooking recipe",
                "metadata": {},
                "provenance": {"source_kind": "test", "source_id": "b"},
            }
        ),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
    scope = VectorStoreScope(tenant_id="t1")
    store.add_records(
        [
            VectorStoreRecord(document=doc, embedding=embedding, vector_id=doc.identity.document_id)
            for doc, embedding in zip(docs, embeddings)
        ],
        scope=scope,
    )

    hits = store.query_hybrid(
        [0.95, 0.05, 0.0],
        "Intergrax RAG",
        scope=scope,
        top_k=2,
        metadata_filter=None,
    )
    assert hits
    assert hits[0].id == "a"
