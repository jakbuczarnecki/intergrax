# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    retrieval_hit_to_chunk,
)

pytestmark = pytest.mark.unit


def test_retrieval_adapter_isolated_preserves_native_and_user_fields() -> None:
    user_metadata = {
        "workspace_id": "workspace-a",
        "source": "policy.md",
        "custom": {"label": "trusted"},
    }
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "document-a", "root_document_id": "document-a"},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": "Native retrieval content.",
            "metadata": user_metadata,
            "provenance": {
                "source_kind": "file",
                "source_id": "source-a",
                "provider_id": "local",
            },
        }
    )
    hit = RetrievalHit(
        document=document,
        score=0.88,
        rank=2,
        channel="dense",
        vector_id="vector-a",
    )

    chunk = retrieval_hit_to_chunk(hit)

    assert chunk.id == "document-a"
    assert chunk.text == "Native retrieval content."
    assert chunk.scope == {"tenant_id": "tenant-a", "namespace": "namespace-a"}
    assert chunk.user_metadata == user_metadata
    assert chunk.metadata == user_metadata
    assert chunk.score == 0.88
    assert chunk.rank == 2
    assert chunk.channel == "dense"
    assert chunk.vector_id == "vector-a"
    assert chunk.provenance["source_id"] == "source-a"
    assert chunk.provenance["provider_id"] == "local"
    assert "document_id" not in chunk.metadata
    assert not hasattr(chunk, "embedding")
