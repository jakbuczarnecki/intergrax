# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Adapt native retrieval hits to legacy RetrievalChunk ABI."""

from __future__ import annotations

import json

from intergrax.rag.retrieval.retrieval_result import RetrievalChunk
from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit


def retrieval_hit_to_chunk(hit: RetrievalHit) -> RetrievalChunk:
    if not isinstance(hit, RetrievalHit):
        raise TypeError("hit must be a RetrievalHit")

    user_metadata = dict(hit.document.metadata)
    provenance = hit.document.provenance.model_dump(mode="json")
    provenance["root_document_id"] = hit.document.identity.root_document_id
    return RetrievalChunk(
        id=hit.document.identity.document_id,
        text=hit.document.content,
        score=hit.score,
        rank=hit.rank,
        channel=hit.channel,
        vector_id=hit.vector_id,
        scope=hit.document.scope.model_dump(mode="json"),
        provenance=provenance,
        user_metadata=dict(user_metadata),
        metadata=dict(user_metadata),
    )


def retrieval_hit_to_json(hit: RetrievalHit) -> str:
    return json.dumps(hit.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["retrieval_hit_to_chunk", "retrieval_hit_to_json"]
