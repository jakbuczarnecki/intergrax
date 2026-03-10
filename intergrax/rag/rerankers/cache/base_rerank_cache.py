# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseRerankCache(ABC):
    """
    Contract for reranker result caches.

    Implementations may use in-memory, Redis, or other storage backends.
    """

    @abstractmethod
    def get(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
    ) -> Optional[List[float]]:
        """
        Retrieve cached rerank scores.
        """
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        *,
        reranker: str,
        query: str,
        texts: List[str],
        scores: List[float],
    ) -> None:
        """
        Store rerank scores in cache.
        """
        raise NotImplementedError