# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Sequence

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.multimedia.video_smart_loader import VideoSmartLoader
from intergrax.rag.document_loaders.contracts.base_document_parser import BaseDocumentParser
from intergrax.rag.document_loaders.contracts.metadata_contract import build_loader_metadata


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

    def load(self, source: str) -> Sequence[ParsedDocumentFragment]:

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

        docs = loader.load()

        result: list[ParsedDocumentFragment] = []

        for i, d in enumerate(docs):

            metadata = build_loader_metadata(
                source=source,
                parser=self.parser_id(),
                position=i,
            )

            metadata.update(d.metadata or {})

            result.append(
                ParsedDocumentFragment(
                    text=d.page_content,
                    metadata=metadata,
                )
            )

        return result
