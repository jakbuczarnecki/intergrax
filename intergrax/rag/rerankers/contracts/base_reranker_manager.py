# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)



class BaseRerankerManager(ABC):

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
        reranker_id: str | None = None,
    ) -> Sequence[RerankerResult]:
        raise NotImplementedError


    def __call__(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
        reranker_id: str | None = None,
    ) -> Sequence[RerankerResult]:

        return self.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
            reranker_id=reranker_id,
        )