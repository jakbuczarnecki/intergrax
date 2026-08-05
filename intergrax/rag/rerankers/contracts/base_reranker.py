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



class BaseReranker(ABC):

    @classmethod
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:
        """
        Re-rank candidate documents for a given query.
        """
        raise NotImplementedError


    def __call__(
        self,
        *,
        query: str,
        candidates: Sequence[RerankerCandidate],
        limit: int | None = None,
    ) -> Sequence[RerankerResult]:

        return self.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )