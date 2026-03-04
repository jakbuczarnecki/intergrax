# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredHTMLLoader

from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)


class UnstructuredHtmlDocumentHandler(BaseDocumentHandler):

    def supports(self, source: str) -> bool:
        s = source.lower()
        return (
            s.endswith(".html")
            or s.endswith(".htm")
        )

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def load(self, source: str) -> Sequence[Document]:
        loader = UnstructuredHTMLLoader(source)
        return loader.load()