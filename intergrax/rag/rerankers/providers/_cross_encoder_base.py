# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from sentence_transformers import CrossEncoder

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker
from intergrax.rag.rerankers.contracts.reranker_types import (
    validate_candidates,
    validate_limit,
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
        self._model_name = model_name or self.DEFAULT_MODEL
        self._max_length = max_length
        self._model: Optional[CrossEncoder] = None        

    def _ensure_cross_encoder(self):
        if self._model is None:
            self._model = CrossEncoder(
                self._model_name,
                max_length=self._max_length,
            )

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        candidates = validate_candidates(candidates)
        validate_limit(limit)
        if not candidates:
            return ()

        if not query.strip():
            return ()
        
        self._ensure_cross_encoder()

        normalized = candidates
        pairs = [(query, c.document.content) for c in normalized]

        scores = self._model.predict(pairs)

        if len(scores) != len(normalized):
            raise ValueError("cross encoder returned an invalid score shape")
        scored = list(zip(normalized, scores))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        selected = scored[:limit] if limit is not None else scored
        return tuple(
            RerankerResult(
                candidate=candidate,
                rerank_score=float(score),
                fusion_score=None,
                rank=rank,
            )
            for rank, (candidate, score) in enumerate(selected)
        )