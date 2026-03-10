# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from intergrax.rag.rerankers.contracts.reranker_types import RerankerResult, Candidates



class BaseReranker(ABC):

    @classmethod
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self,
        *,
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:
        """
        Re-rank candidate documents for a given query.
        """
        raise NotImplementedError


    def __call__(
        self,
        *,
        query: Optional[str],
        candidates: Candidates,
        limit: Optional[int] = None,
    ) -> List[RerankerResult]:

        return self.rerank(
            query=query,
            candidates=candidates,
            limit=limit,
        )