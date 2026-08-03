# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_core.documents import Document

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.audio_smart_document_handler import (
    AudioSmartDocumentHandler,
)
pytestmark = pytest.mark.integration


class _FakeWhisperBackend:

    def parser_id(self) -> str:
        return "whisper"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> list[ParsedDocumentFragment]:
        return [
            ParsedDocumentFragment(
                text="dummy audio transcript",
                metadata={"source": "audio_test"},
            )
        ]


def test_audio_handler_supports_extensions():

    handler = AudioSmartDocumentHandler()

    assert handler.supports("file.wav") is True
    assert handler.supports("file.mp3") is True
    assert handler.supports("file.flac") is True
    assert handler.supports("file.txt") is False


def test_audio_handler_confidence():

    handler = AudioSmartDocumentHandler()

    assert handler.confidence("file.wav") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_audio_handler_builds_parser():

    handler = AudioSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) >= 1
    assert parsers[0].parser_id() == "whisper"


def test_audio_handler_load(monkeypatch, tmp_path: Path):

    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.audio_smart_parser.resolve_document_parser",
        lambda slug, **kwargs: _FakeWhisperBackend(),
    )

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")

    handler = AudioSmartDocumentHandler()

    docs = handler.load(str(audio_path))

    assert docs
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "dummy audio transcript"