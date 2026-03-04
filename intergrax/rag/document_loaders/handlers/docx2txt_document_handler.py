# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader

from intergrax.rag.document_loaders.config.document_loader_config import DEFAULT_BUILTIN_HANDLER_CONFIDENCE
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler


class Docx2TxtDocumentHandler(BaseDocumentHandler):

    def supports(self, source: str) -> bool:
        return source.lower().endswith(".docx")

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def load(self, source: str) -> Sequence[Document]:
        loader = Docx2txtLoader(source)
        return loader.load()