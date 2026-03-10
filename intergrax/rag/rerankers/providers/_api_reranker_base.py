# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    Candidates,
    RerankerCandidate,
    RerankerResult,
)


class _APIRerankerBase(BaseReranker, ABC):

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

        texts: List[str] = [c.text for c in normalized]

        scores = self._score(query, texts)

        results: List[RerankerResult] = []

        for candidate, score in zip(normalized, scores):

            results.append(
                RerankerResult(
                    candidate=candidate,
                    rerank_score=score,
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

    @abstractmethod
    def _score(
        self,
        query: str,
        texts: List[str],
    ) -> List[float]:
        ...