# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge Integration Library document parsers into RAG document_loaders."""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.document_metadata_key import DocumentMetadataKey
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata


class CatalogDocumentParser(BaseDocumentParser):
    """Wrap a catalog ``DocumentParser`` as a RAG ``BaseDocumentParser``."""

    def __init__(self, backend: DocumentParser) -> None:
        self._backend = backend

    @classmethod
    def parser_id(cls) -> str:
        return "catalog"

    def parser_id_instance(self) -> str:
        return self._backend.parser_id()

    def is_available(self) -> bool:
        return self._backend.is_available()

    def load(self, source: str) -> Sequence[Document]:
        fragments = self._backend.parse_file(source)
        parser_id = self._backend.parser_id()
        documents: list[Document] = []
        for index, fragment in enumerate(fragments):
            metadata = build_loader_metadata(
                source=source,
                parser=parser_id,
                position=index,
            )
            metadata.update(fragment.metadata)
            if fragment.native_handle is not None:
                metadata[DocumentMetadataKey.DOCLING_DOCUMENT_META] = fragment.native_handle
            documents.append(
                Document(page_content=fragment.text, metadata=metadata)
            )
        return documents
