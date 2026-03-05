# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path
from typing import List

from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
)


class DocumentHandlerRegistry:
    """
    Registry responsible for selecting the most appropriate document handler
    for a given source file.

    Handler selection uses a two-step process:

    1. Capability filtering via `supports()`
    2. Negotiation via `confidence()`
    """

    def __init__(self) -> None:
        self._handlers: List[BaseDocumentHandler] = []


    def register(self, handler: BaseDocumentHandler) -> None:
        """
        Register a new document handler.
        """
        self._handlers.append(handler)


    def resolve(self, source: str) -> BaseDocumentHandler:
        """
        Resolve the best handler for the given document source.

        Parameters
        ----------
        source : str
            Source URI (path, http url, s3 uri, etc.)

        Returns
        -------
        BaseDocumentHandler
            Selected handler.

        Raises
        ------
        RuntimeError
            If no handler supports the document.
        """

        candidates: List[BaseDocumentHandler] = []

        for handler in self._handlers:
            if handler.supports(source):
                candidates.append(handler)

        if not candidates:
            raise RuntimeError(f"No document handler available for: {source}")

        best_handler = candidates[0]
        best_score = best_handler.confidence(source)

        for handler in candidates[1:]:
            score = handler.confidence(source)
            if score > best_score:
                best_score = score
                best_handler = handler

        return best_handler