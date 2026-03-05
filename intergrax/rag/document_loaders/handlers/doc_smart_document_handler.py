# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List


from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.contracts.base_document_handler import (
    BaseDocumentHandler,
)
from intergrax.rag.document_loaders.contracts.base_document_parser import (
    BaseDocumentParser,
)
from intergrax.rag.document_loaders.parsers.doc_smart_parser import EXTRACTION_STRATEGY, DocSmartParser

class DocSmartDocumentHandler(BaseDocumentHandler):

    def __init__(self, extraction_strategy: EXTRACTION_STRATEGY = "auto") -> None:
        self._extraction_strategy = extraction_strategy

    def supports(self, source: str) -> bool:
        s = source.lower()
        return s.endswith(".docx") or s.endswith(".doc")

    def confidence(self, source: str) -> float:
        return GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence

    def build_parsers(self) -> List[BaseDocumentParser]:
        return [
            DocSmartParser(strategy=self._extraction_strategy)
        ]
