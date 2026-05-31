# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from intergrax.integrations.registry.slugs import IntegrationSlug
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata
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
        return resolve_document_parser(IntegrationSlug.WHISPER, **self._options).is_available()

    def load(self, source: str) -> Sequence[Document]:
        backend = resolve_document_parser(IntegrationSlug.WHISPER, **self._options)
        docs = CatalogDocumentParser(backend).load(source)
        result: list[Document] = []
        for i, doc in enumerate(docs):
            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )
            metadata.update(doc.metadata or {})
            result.append(Document(page_content=doc.page_content, metadata=metadata))
        return result
