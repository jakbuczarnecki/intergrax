# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable

from intergrax.rag.retrievers.contracts.base_retriever import BaseRetriever


DEFAULT_RETRIEVER_ID : str = "default"

class RetrieverRegistry:
    """
    Registry for retriever strategies.
    """

    def __init__(self, retrievers: Iterable[BaseRetriever] | None = None):
        self._retrievers: Dict[str, BaseRetriever] = {}

        if retrievers:
            for retriever in retrievers:
                self.register(retriever)

    def register(self, retriever: BaseRetriever) -> None:

        name = retriever.name()

        if name in self._retrievers:
            raise ValueError(
                f"Retriever already registered: {name}"
            )

        self._retrievers[name] = retriever

    def get(self, name: str) -> BaseRetriever:

        if name is None or name == DEFAULT_RETRIEVER_ID:
            name = self.default_retriever()
        
        retriever = self._retrievers.get(name)

        if retriever is None:
            raise RuntimeError(
                f"Retriever not registered: {name}"
            )

        return retriever

    def default_retriever(self) -> str:
        """
        Returns identifier of the default retriever.
        """

        if not self._retrievers:
            raise RuntimeError("No retrievers registered.")

        return next(iter(self._retrievers.keys()))