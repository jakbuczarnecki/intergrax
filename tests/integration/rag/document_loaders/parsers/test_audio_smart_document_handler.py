# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.audio_smart_document_handler import (
    AudioSmartDocumentHandler,
)
pytestmark = pytest.mark.integration


class _DummyAudioLoader:

    def __init__(self, *args, **kwargs):
        pass

    def load(self):

        return [
            Document(
                page_content="dummy audio transcript",
                metadata={"source": "audio_test"}
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
    assert parsers[0].parser_id() == "audio_smart"


def test_audio_handler_load(monkeypatch, tmp_path: Path):

    from intergrax.rag.document_loaders.parsers import audio_smart_parser

    monkeypatch.setattr(
        audio_smart_parser,
        "AudioSmartLoader",
        _DummyAudioLoader
    )

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio")

    handler = AudioSmartDocumentHandler()

    docs = handler.load(str(audio_path))

    assert docs
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "dummy audio transcript"