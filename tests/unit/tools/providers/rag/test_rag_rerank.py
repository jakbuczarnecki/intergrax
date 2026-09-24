# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate, RerankerResult
from intergrax.rag.rerankers.engine.reranker_engine import RerankerEngine
from intergrax.rag.rerankers.registry.reranker_registry import RerankerRegistry
from intergrax.rag.rerankers.re_ranker_manager import ReRankerManager
from intergrax.tools.providers.rag.rerank_contracts import RagRerankChunkInput, RagRerankInput
from intergrax.tools.providers.rag.rerank_service import rag_rerank
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _QueryOverlapReranker(BaseReranker):
    """Deterministic lexical overlap — hermetic unit qualification only."""

    def name(self) -> str:
        return "query_overlap"

    def rerank(
        self,
        *,
        query: str | None,
        candidates: list[RerankerCandidate],
        limit: int | None = None,
    ) -> list[RerankerResult]:
        query_terms = {term for term in (query or "").lower().split() if term}
        scored: list[tuple[float, RerankerCandidate]] = []
        for candidate in candidates:
            text_terms = set(candidate.document.content.lower().split())
            overlap = float(len(query_terms & text_terms))
            scored.append((overlap, candidate))
        scored.sort(key=lambda item: (-item[0], item[1].original_rank))
        if limit is not None:
            scored = scored[:limit]
        return [
            RerankerResult(
                candidate=candidate,
                rerank_score=score,
                fusion_score=None,
                rank=rank,
            )
            for rank, (score, candidate) in enumerate(scored, start=1)
        ]


def _hermetic_reranker_manager() -> ReRankerManager:
    registry = RerankerRegistry([_QueryOverlapReranker()])
    return ReRankerManager(engine=RerankerEngine(registry))


def test_rag_rerank_orders_candidates() -> None:
    ctx = ToolWiringContext(reranker_manager=_hermetic_reranker_manager())
    out = rag_rerank(
        ctx,
        RagRerankInput(
            query="project budget",
            chunks=[
                RagRerankChunkInput(id="a", text="unrelated note"),
                RagRerankChunkInput(id="b", text="project budget summary"),
            ],
            top_n=2,
        ),
    )
    assert out.total == 2
    assert out.chunks[0].id == "b"
    assert out.chunks[1].id == "a"
