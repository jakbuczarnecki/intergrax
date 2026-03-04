# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence
from pathlib import Path

from langchain_core.documents import Document

from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.config.document_loader_config import DEFAULT_BUILTIN_HANDLER_CONFIDENCE

from intergrax.multimedia.audio_smart_loader import AudioSmartLoader


class AudioSmartDocumentHandler(BaseDocumentHandler):

    _SUPPORTED_EXTENSIONS = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".opus",
        ".aac",
        ".wma",
        ".aiff",
        ".aif",
        ".mka",
    }

    def __init__(
        self,
        *,
        out_dir: str | None = None,
        whisper_model: str = "medium",
        whisper_language: str | None = None,
        translate: bool = True,
    ):
        self._out_dir = out_dir
        self._whisper_model = whisper_model
        self._whisper_language = whisper_language
        self._translate = translate

    def supports(self, source: str) -> bool:
        source_lower = source.lower()
        return any(source_lower.endswith(ext) for ext in self._SUPPORTED_EXTENSIONS)

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def build_parsers(self) -> List[BaseDocumentParser]:

        return [
            AudioSmartParser(
                out_dir=self._out_dir,
                whisper_model=self._whisper_model,
                whisper_language=self._whisper_language,
                translate=self._translate,
            )
        ]


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

        return loader.load()