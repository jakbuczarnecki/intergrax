# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    Candidates,
    RerankerCandidate,
    RerankerResult,
)


class _CrossEncoderBaseReranker(BaseReranker):

    DEFAULT_MODEL: str

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        max_length: int = 512,
    ) -> None:

        if model_name is None:
            model_name = self.DEFAULT_MODEL

        self._model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        if not candidates:
            return []

        if query is None or not query.strip():
            return []

        normalized: List[RerankerCandidate] = []

        if isinstance(candidates[0], Document):

            for d in candidates:

                normalized.append(
                    RerankerCandidate(
                        id=None,
                        text=d.page_content or "",
                        metadata=dict(d.metadata or {}),
                        original_score=None,
                    )
                )

        else:
            normalized = list(candidates)

        pairs = [(query, c.text) for c in normalized]

        scores = self._model.predict(pairs)

        results: List[RerankerResult] = []

        for c, score in zip(normalized, scores):

            results.append(
                RerankerResult(
                    candidate=c,
                    rerank_score=float(score),
                    fusion_score=None,
                    rank=0,
                )
            )

        results.sort(
            key=lambda r: r.rerank_score,
            reverse=True,
        )

        if limit:
            results = results[:limit]

        for i, r in enumerate(results, start=1):
            r.rank = i

        return results