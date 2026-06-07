# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.rerankers.bootstrap.reranker_bootstrap import create_default_reranker_manager
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate
from intergrax.tools.providers.rag.rerank_contracts import RagRerankChunkOutput, RagRerankInput, RagRerankOutput
from intergrax.tools.registry.wiring import ToolWiringContext

RAG_RERANK_TOOL_ID = "rag.rerank"


def rag_rerank(ctx: ToolWiringContext, params: RagRerankInput) -> RagRerankOutput:
    profile = ctx.rag_profile or RagProfile()
    reranker_manager = ctx.reranker_manager
    if reranker_manager is None and profile.enable_rerank:
        reranker_manager = create_default_reranker_manager()
    if reranker_manager is None:
        raise RuntimeError("reranker_manager_not_configured")

    candidates = [
        RerankerCandidate(
            id=chunk.id or None,
            text=chunk.text,
            metadata={key: value for key, value in chunk.metadata.items() if value is not None},
            original_score=chunk.score,
        )
        for chunk in params.chunks
    ]
    results = reranker_manager.rerank(query=params.query.strip(), candidates=candidates, limit=params.top_n)
    reranker_id = profile.reranker_id or "default"
    output_chunks = [
        RagRerankChunkOutput(
            id=item.candidate.id or "",
            text=item.candidate.text,
            score=item.rerank_score,
            rank=item.rank,
            metadata=dict(item.candidate.metadata),
        )
        for item in results
    ]
    return RagRerankOutput(
        query=params.query.strip(),
        chunks=output_chunks,
        reranker_id=reranker_id,
        total=len(output_chunks),
    )
