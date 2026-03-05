# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser

class HtmlSmartParser(BaseDocumentParser):

    @classmethod
    def parser_id(cls) -> str:
        return "html_smart"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        loader = UnstructuredHTMLLoader(source)

        return loader.load()