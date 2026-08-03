# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser


class HtmlSmartParser(BaseDocumentParser):
    @classmethod
    def parser_id(cls) -> str:
        return "unstructured"

    def is_available(self) -> bool:
        return resolve_document_parser("unstructured").is_available()

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        backend = resolve_document_parser("unstructured")
        return CatalogDocumentParser(backend).load(source)
