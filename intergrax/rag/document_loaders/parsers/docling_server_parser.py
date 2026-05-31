# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser


class DoclingServerParser(BaseDocumentParser):
    """Deprecated alias — use ``DoclingLocalParser`` with mode=server via integration profile."""

    @classmethod
    def parser_id(cls) -> str:
        return "docling.server"

    def is_available(self) -> bool:
        backend = resolve_document_parser(IntegrationSlug.DOCLING, mode="server")
        return backend.is_available()

    def load(self, source: str) -> Sequence[Document]:
        backend = resolve_document_parser(IntegrationSlug.DOCLING, mode="server")
        return CatalogDocumentParser(backend).load(source)
