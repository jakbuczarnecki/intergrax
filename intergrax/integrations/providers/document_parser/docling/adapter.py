# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import DocumentParser, ParsedDocumentFragment
from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig, DoclingMode
from intergrax.integrations.providers.document_parser.docling.opens import parse_docling_file


class _DoclingDocumentParser:
    """Catalog ``DocumentParser`` for Docling local or server mode."""

    def __init__(self, config: DoclingIntegrationConfig) -> None:
        self._config = config

    def parser_id(self) -> str:
        if self._config.mode is DoclingMode.SERVER:
            return "docling.server"
        return "docling.local"

    def is_available(self) -> bool:
        return self._config.mode in {DoclingMode.LOCAL, DoclingMode.SERVER}

    def parse_file(self, source: str) -> Sequence[ParsedDocumentFragment]:
        return parse_docling_file(self._config, source)
