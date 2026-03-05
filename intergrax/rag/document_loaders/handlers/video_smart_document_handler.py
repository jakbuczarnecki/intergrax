# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)
from intergrax.rag.document_loaders.parsers.video_smart_parser import VideoSmartParser


class VideoSmartDocumentHandler(BaseDocumentHandler):

    _SUPPORTED_EXTENSIONS = {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".webm",
        ".m4v",
        ".flv",
        ".wmv",
        ".ts",
        ".3gp",
        ".ogv",
    }

    def __init__(
        self,
        *,
        out_dir: str | None = None,
        frames_subdir: str = "frames",
        meta_subdir: str = "video_meta",
        transcribe_if_missing: bool = True,
        whisper_model_size: str = "base",
        whisper_language: str | None = None,
        frame_target_height: int = 350,
    ):

        self._out_dir = out_dir
        self._frames_subdir = frames_subdir
        self._meta_subdir = meta_subdir
        self._transcribe_if_missing = transcribe_if_missing
        self._whisper_model_size = whisper_model_size
        self._whisper_language = whisper_language
        self._frame_target_height = frame_target_height

    def supports(self, source: str) -> bool:

        source_lower = source.lower()

        return any(source_lower.endswith(ext) for ext in self._SUPPORTED_EXTENSIONS)

    def confidence(self, source: str) -> float:
        return DEFAULT_BUILTIN_HANDLER_CONFIDENCE

    def build_parsers(self) -> List[BaseDocumentParser]:

        return [
            VideoSmartParser(
                out_dir=self._out_dir,
                frames_subdir=self._frames_subdir,
                meta_subdir=self._meta_subdir,
                transcribe_if_missing=self._transcribe_if_missing,
                whisper_model_size=self._whisper_model_size,
                whisper_language=self._whisper_language,
                frame_target_height=self._frame_target_height,
            )
        ]