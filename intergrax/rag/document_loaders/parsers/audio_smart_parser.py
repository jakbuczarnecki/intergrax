# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.integration.catalog_parser import CatalogDocumentParser
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser


class AudioSmartParser(BaseDocumentParser):

    def __init__(
        self,
        *,
        out_dir: str | None,
        whisper_model: str,
        whisper_language: str | None,
        translate: bool,
    ):
        self._options = {
            "model": whisper_model,
            "language": whisper_language or "en",
            "translate": translate,
            "out_dir": out_dir,
        }

    @classmethod
    def parser_id(cls) -> str:
        return "whisper"

    def is_available(self) -> bool:
        return resolve_document_parser("whisper", **self._options).is_available()

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:
        backend = resolve_document_parser("whisper", **self._options)
        return CatalogDocumentParser(backend).load(source)
