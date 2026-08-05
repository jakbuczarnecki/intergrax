# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate


def _document(document_id: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        schema_version=1,
        identity={
            "document_id": document_id,
            "root_document_id": document_id,
        },
        scope={"tenant_id": "test-tenant", "namespace": "rerank"},
        content={
            "1": "Paris is the capital of France",
            "2": "Bananas are yellow fruits",
            "3": "Berlin is the capital of Germany",
        }[document_id],
        metadata={"fixture": True},
        provenance={"source_kind": "test", "source_id": document_id},
    )


def candidates() -> list[RerankerCandidate]:

    return [
        RerankerCandidate(
            document=_document("1"),
            original_score=0.1,
            original_rank=0,
            channel="fixture",
            vector_id="v1",
        ),
        RerankerCandidate(
            document=_document("2"),
            original_score=0.1,
            original_rank=1,
            channel="fixture",
            vector_id="v2",
        ),
        RerankerCandidate(
            document=_document("3"),
            original_score=0.1,
            original_rank=2,
            channel="fixture",
            vector_id="v3",
        ),
    ]
