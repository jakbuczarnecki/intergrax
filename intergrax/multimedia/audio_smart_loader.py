# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pathlib import Path

from intergrax.knowledge.contracts import KnowledgeDocument


class AudioSmartLoader:
    """Delegates audio / YouTube ingestion to the whisper integration catalog backend."""

    def __init__(
        self,
        path: str,
        *,
        out_dir: str | Path | None = None,
        audio_format: str = "mp3",
        whisper_model: str = "medium",
        whisper_language: str = "en",
        translate: bool = True,
    ):
        self.path = path
        self._options = {
            "model": whisper_model,
            "language": whisper_language,
            "translate": translate,
            "out_dir": str(out_dir or "./audio_downloads"),
            "audio_format": audio_format,
        }

    def load(self) -> list[KnowledgeDocument]:
        from intergrax.rag.document_loaders.integration.catalog_parser import (
            CatalogDocumentParser,
        )
        from intergrax.rag.document_loaders.integration.resolver import (
            resolve_document_parser,
        )

        backend = resolve_document_parser("whisper", **self._options)
        documents = list(CatalogDocumentParser(backend).load(self.path))
        if not all(isinstance(document, KnowledgeDocument) for document in documents):
            raise TypeError("audio parser must return KnowledgeDocument values")
        return documents
