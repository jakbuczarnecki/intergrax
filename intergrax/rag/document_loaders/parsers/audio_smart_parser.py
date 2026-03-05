# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence
from pathlib import Path

from langchain_core.documents import Document

from intergrax.multimedia.audio_smart_loader import AudioSmartLoader
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata

class AudioSmartParser(BaseDocumentParser):

    def __init__(
        self,
        *,
        out_dir: str | None,
        whisper_model: str,
        whisper_language: str | None,
        translate: bool,
    ):
        self._out_dir = out_dir
        self._whisper_model = whisper_model
        self._whisper_language = whisper_language
        self._translate = translate

    @classmethod
    def parser_id(cls) -> str:
        return "audio_smart"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        path = Path(source)
        ext = path.suffix.lower()

        audio_format = ext.lstrip(".")

        loader = AudioSmartLoader(
            path=source,
            out_dir=self._out_dir,
            audio_format=audio_format,
            whisper_model=self._whisper_model,
            whisper_language=self._whisper_language,
            translate=self._translate,
        )

        docs = loader.load()

        result: list[Document] = []

        for i, d in enumerate(docs):

            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )

            metadata.update(d.metadata or {})

            result.append(
                Document(
                    page_content=d.page_content,
                    metadata=metadata,
                )
            )

        return result