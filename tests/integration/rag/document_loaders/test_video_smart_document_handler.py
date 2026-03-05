# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.video_smart_document_handler import (
    VideoSmartDocumentHandler,
)

pytestmark = pytest.mark.integration


class _DummyVideoLoader:

    def __init__(self, *args, **kwargs):
        pass

    def load(self):

        return [
            Document(
                page_content="dummy video transcript",
                metadata={"source": "video_test"}
            )
        ]


def test_video_handler_supports_extensions():

    handler = VideoSmartDocumentHandler()

    assert handler.supports("video.mp4") is True
    assert handler.supports("video.mkv") is True
    assert handler.supports("video.webm") is True
    assert handler.supports("video.txt") is False


def test_video_handler_confidence():

    handler = VideoSmartDocumentHandler()

    assert handler.confidence("video.mp4") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_video_handler_builds_parser():

    handler = VideoSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) == 1
    assert parsers[0].parser_id() == "video_smart"


def test_video_handler_load(monkeypatch, tmp_path: Path):

    from intergrax.rag.document_loaders.parsers import video_smart_parser

    monkeypatch.setattr(
        video_smart_parser,
        "VideoSmartLoader",
        _DummyVideoLoader
    )

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"fake video")

    handler = VideoSmartDocumentHandler()

    docs = handler.load(str(video_path))

    assert docs
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "dummy video transcript"