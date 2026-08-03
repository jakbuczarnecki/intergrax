# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser


class DoclingLocalParser(BaseDocumentParser):
    @classmethod
    def parser_id(cls) -> str:
        return "docling.local"

    def is_available(self) -> bool:
        backend = resolve_document_parser("docling", mode="local")
        return backend.is_available()

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        backend = resolve_document_parser("docling", mode="local")
        return CatalogDocumentParser(backend).load(source)


class DoclingServerParser(BaseDocumentParser):
    @classmethod
    def parser_id(cls) -> str:
        return "docling.server"

    def is_available(self) -> bool:
        backend = resolve_document_parser("docling", mode="server")
        return backend.is_available()

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        backend = resolve_document_parser("docling", mode="server")
        return CatalogDocumentParser(backend).load(source)
