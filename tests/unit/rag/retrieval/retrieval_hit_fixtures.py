# © Artur Czarnecki. All rights reserved.

"""Shared RetrievalHit stubs for retrieval service unit tests."""

from __future__ import annotations

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit


def stub_knowledge_document(
    *,
    document_id: str = "doc-1",
    content: str = "stub content",
    metadata: dict[str, object] | None = None,
) -> KnowledgeDocument:
    return KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": document_id, "root_document_id": document_id},
            "scope": {"tenant_id": "tenant-a", "namespace": "namespace-a"},
            "content": content,
            "metadata": dict(metadata or {}),
            "provenance": {"source_kind": "test", "source_id": document_id},
        }
    )


def stub_retrieval_hit(
    *,
    content: str = "stub content",
    document_id: str = "doc-1",
    score: float = 0.9,
    rank: int = 0,
    channel: str = "dense",
    metadata: dict[str, object] | None = None,
) -> RetrievalHit:
    return RetrievalHit(
        document=stub_knowledge_document(
            document_id=document_id,
            content=content,
            metadata=metadata,
        ),
        score=score,
        rank=rank,
        channel=channel,
        vector_id=f"vec-{document_id}",
    )
