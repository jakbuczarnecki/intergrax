# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable

from intergrax.rag.rerankers.contracts.base_reranker import BaseReranker


class RerankerRegistry:
    """
    Registry for reranker strategies.
    """

    def __init__(self, rerankers: Iterable[BaseReranker] | None = None):

        self._rerankers: Dict[str, BaseReranker] = {}

        if rerankers:
            for reranker in rerankers:
                self.register(reranker)

    def register(self, reranker: BaseReranker) -> None:

        name = reranker.name()

        if name in self._rerankers:
            raise ValueError(
                f"Reranker already registered: {name}"
            )

        self._rerankers[name] = reranker

    def get(self, name: str) -> BaseReranker:

        reranker = self._rerankers.get(name)

        if reranker is None:
            raise RuntimeError(
                f"Reranker not registered: {name}"
            )

        return reranker

    def default_reranker(self) -> str:
        """
        Returns identifier of the default reranker.
        """

        if not self._rerankers:
            raise RuntimeError(
                "No rerankers registered."
            )

        return next(iter(self._rerankers.keys()))