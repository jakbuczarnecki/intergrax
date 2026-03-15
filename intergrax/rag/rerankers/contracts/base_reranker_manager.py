# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from intergrax.rag.rerankers.contracts.reranker_types import RerankerCandidate, RerankerResult, Candidates



class BaseRerankerManager(ABC):

    @abstractmethod
    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: List[RerankerCandidate],
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:
        raise NotImplementedError


    def __call__(
        self,
        *,
        query: Optional[str],
        candidates: List[RerankerCandidate],
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        return self.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )