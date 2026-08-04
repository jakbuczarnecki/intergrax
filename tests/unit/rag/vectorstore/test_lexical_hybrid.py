# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore

pytestmark = pytest.mark.unit


def test_inmemory_query_hybrid_prefers_keyword_match() -> None:
    store = InMemoryVectorStore(tenant_id="t1")
    docs = [
        Document(page_content="Intergrax harness RAG pipeline", metadata={}),
        Document(page_content="unrelated cooking recipe", metadata={}),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]
    store.add_documents(docs, embeddings, ids=["a", "b"])

    hits = store.query_hybrid(
        [0.95, 0.05, 0.0],
        "Intergrax RAG",
        top_k=2,
        metadata_filter=None,
    )
    assert hits
    assert hits[0].id == "a"
