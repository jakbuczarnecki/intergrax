# © Artur Czarnecki. All rights reserved.

"""Hybrid retrieval orchestrator — vector + keyword + graph (Phase MEM-DEPTH-4.3)."""

from __future__ import annotations

from typing import Sequence

from intergrax.runtime.architecture.hybrid_retrieval import (
    ChannelRetrievalHit,
    HybridRetrievalRequest,
    HybridRetrievalResult,
    RetrievalChannel,
    execute_hybrid_retrieval,
)


def orchestrate_hybrid_retrieval(
    *,
    query_id: str,
    vector_document_ids: Sequence[tuple[str, float]],
    keyword_document_ids: Sequence[tuple[str, float]] = (),
    graph_document_ids: Sequence[tuple[str, float]] = (),
    top_k: int = 8,
) -> HybridRetrievalResult:
    """Rank a unified result set from channel-specific hits."""
    request = HybridRetrievalRequest(
        query_id=query_id,
        top_k=top_k,
        vector_hits=[
            ChannelRetrievalHit(
                channel=RetrievalChannel.VECTOR,
                document_id=document_id,
                score=score,
            )
            for document_id, score in vector_document_ids
        ],
        keyword_hits=[
            ChannelRetrievalHit(
                channel=RetrievalChannel.KEYWORD,
                document_id=document_id,
                score=score,
            )
            for document_id, score in keyword_document_ids
        ],
        graph_hits=[
            ChannelRetrievalHit(
                channel=RetrievalChannel.GRAPH,
                document_id=document_id,
                score=score,
            )
            for document_id, score in graph_document_ids
        ],
    )
    return execute_hybrid_retrieval(request)
