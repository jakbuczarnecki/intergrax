# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import List, Optional

RerankIdentityKey = tuple[str, str | None, str | None, str, str | None]


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
        identity_keys: Sequence[RerankIdentityKey] | None = None,
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
        identity_keys: Sequence[RerankIdentityKey] | None = None,
        scores: List[float],
    ) -> None:
        """
        Store rerank scores in cache.
        """
        raise NotImplementedError