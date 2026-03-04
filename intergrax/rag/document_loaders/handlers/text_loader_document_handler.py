# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)


class TextLoaderDocumentHandler(BaseDocumentHandler):

    def supports(self, source: str) -> bool:
        s = source.lower()
        return (
            s.endswith(".txt")
            or s.endswith(".md")
            or s.endswith(".markdown")
        )

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def load(self, source: str) -> Sequence[Document]:
        loader = TextLoader(source, autodetect_encoding=True)
        return loader.load()