# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from intergrax.rag.rerankers.contracts.reranker_types import (
    RerankerCandidate,
    RerankerResult,
)

if TYPE_CHECKING:
    from intergrax.rag.embedding.contracts.base_embedding_manager import (
        BaseEmbeddingManager,
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


class BaseRerankerPlugin(ABC):
    """Typed construction contract for dependency-aware reranker plugins."""

    @classmethod
    @abstractmethod
    def create(cls, *, embedding_manager: BaseEmbeddingManager) -> BaseReranker:
        """Construct a reranker from the normal RAG embedding dependency."""
        raise NotImplementedError