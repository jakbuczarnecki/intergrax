# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List


from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)
from intergrax.rag.document_loaders.parsers.html_smart_parser import HtmlSmartParser


class HtmlSmartDocumentHandler(BaseDocumentHandler):

    def supports(self, source: str) -> bool:

        s = source.lower()

        return (
            s.endswith(".html")
            or s.endswith(".htm")
        )

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def build_parsers(self) -> List[BaseDocumentParser]:

        return [
            HtmlSmartParser()
        ]