# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Sequence

from langchain_core.documents import Document

from intergrax.multimedia.video_smart_loader import VideoSmartLoader
from intergrax.rag.document_loaders.contracts.base_document_handler import BaseDocumentHandler
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.config.document_loader_config import (
    DEFAULT_BUILTIN_HANDLER_CONFIDENCE,
)


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


class VideoSmartParser(BaseDocumentParser):

    def __init__(
        self,
        *,
        out_dir: str | None,
        frames_subdir: str,
        meta_subdir: str,
        transcribe_if_missing: bool,
        whisper_model_size: str,
        whisper_language: str | None,
        frame_target_height: int,
    ):

        self._out_dir = out_dir
        self._frames_subdir = frames_subdir
        self._meta_subdir = meta_subdir
        self._transcribe_if_missing = transcribe_if_missing
        self._whisper_model_size = whisper_model_size
        self._whisper_language = whisper_language
        self._frame_target_height = frame_target_height

    @classmethod
    def parser_id(cls) -> str:
        return "video_smart"

    def is_available(self) -> bool:
        return True

    def load(self, source: str) -> Sequence[Document]:

        loader = VideoSmartLoader(
            source,
            out_dir=self._out_dir,
            frames_subdir=self._frames_subdir,
            meta_subdir=self._meta_subdir,
            transcribe_if_missing=self._transcribe_if_missing,
            whisper_model_size=self._whisper_model_size,
            whisper_language=self._whisper_language,
            frame_target_height=self._frame_target_height,
        )

        return loader.load()