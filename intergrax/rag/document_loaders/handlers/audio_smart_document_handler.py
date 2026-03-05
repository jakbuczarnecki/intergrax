# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List


from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser

from intergrax.rag.document_loaders.parsers.audio_smart_parser import AudioSmartParser


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
        return GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence

    def build_parsers(self) -> List[BaseDocumentParser]:

        return [
            AudioSmartParser(
                out_dir=self._out_dir,
                whisper_model=self._whisper_model,
                whisper_language=self._whisper_language,
                translate=self._translate,
            )
        ]
