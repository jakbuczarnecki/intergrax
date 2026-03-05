# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List


from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG, DoclingMode
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.parsers.docling_local_parser import DoclingLocalParser
from intergrax.rag.document_loaders.parsers.docling_server_parser import DoclingServerParser
from intergrax.rag.document_loaders.parsers.html_smart_parser import HtmlSmartParser


class HtmlSmartDocumentHandler(BaseDocumentHandler):

    def supports(self, source: str) -> bool:

        s = source.lower()

        return (
            s.endswith(".html")
            or s.endswith(".htm")
        )

    def confidence(self, source: str) -> float:
        return GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence

    def build_parsers(self) -> List[BaseDocumentParser]:

        parsers: List[BaseDocumentParser] = []

        mode = GLOBAL_DOCUMENT_LOADER_CONFIG.docling_mode

        if mode is DoclingMode.LOCAL:
            parsers.append(DoclingLocalParser())

        elif mode is DoclingMode.SERVER:
            parsers.append(DoclingServerParser())

        parsers.append(HtmlSmartParser())

        return parsers