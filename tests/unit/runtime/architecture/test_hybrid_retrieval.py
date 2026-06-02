from __future__ import annotations

from intergrax.runtime.architecture.hybrid_retrieval import (
    ChannelRetrievalHit,
    HybridRetrievalRequest,
    RetrievalChannel,
    execute_hybrid_retrieval,
)


def test_hybrid_retrieval_merges_channels_with_score_fusion() -> None:
    result = execute_hybrid_retrieval(
        HybridRetrievalRequest(
            query_id="q1",
            vector_hits=[
                ChannelRetrievalHit(
                    channel=RetrievalChannel.VECTOR,
                    document_id="doc-1",
                    score=0.9,
                )
            ],
            keyword_hits=[
                ChannelRetrievalHit(
                    channel=RetrievalChannel.KEYWORD,
                    document_id="doc-1",
                    score=0.4,
                ),
                ChannelRetrievalHit(
                    channel=RetrievalChannel.KEYWORD,
                    document_id="doc-2",
                    score=0.8,
                ),
            ],
            graph_hits=[],
            top_k=2,
        )
    )
    assert result.merged_document_ids[0] == "doc-1"
