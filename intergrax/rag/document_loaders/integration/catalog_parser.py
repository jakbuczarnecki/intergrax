# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Bridge Integration Library document parsers into RAG document_loaders."""

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import (
    DocumentParser,
    ParsedDocumentFragment,
)
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
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

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        fragments = self._backend.parse_file(source)
        parser_id = self._backend.parser_id()
        result: list[ParsedDocumentFragment] = []
        for index, fragment in enumerate(fragments):
            metadata = build_loader_metadata(
                source=source,
                parser=parser_id,
                position=index,
            )
            metadata.update(fragment.metadata)
            result.append(
                ParsedDocumentFragment(
                    text=fragment.text,
                    metadata=metadata,
                    native_handle=fragment.native_handle,
                )
            )
        return result
