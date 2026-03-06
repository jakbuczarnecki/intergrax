# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.image_smart_document_handler import (
    ImageSmartDocumentHandler,
)

pytestmark = pytest.mark.integration


class _DummyImageLoader:

    def __init__(self, *args, **kwargs):
        pass

    def load(self):

        return [
            Document(
                page_content="dummy image text",
                metadata={"source": "image_test"}
            )
        ]


def test_image_handler_supports_extensions():

    handler = ImageSmartDocumentHandler()

    assert handler.supports("file.jpg") is True
    assert handler.supports("file.png") is True
    assert handler.supports("file.webp") is True
    assert handler.supports("file.txt") is False


def test_image_handler_confidence():

    handler = ImageSmartDocumentHandler()

    assert handler.confidence("file.jpg") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_image_handler_builds_parser():

    handler = ImageSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) == 1
    assert parsers[0].parser_id() == "image_smart"


def test_image_handler_load(monkeypatch, tmp_path: Path):

    from intergrax.rag.document_loaders.parsers import image_smart_parser

    monkeypatch.setattr(
        image_smart_parser,
        "ImageSmartLoader",
        _DummyImageLoader
    )

    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake image")

    handler = ImageSmartDocumentHandler()

    docs = handler.load(str(image_path))

    assert docs
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "dummy image text"