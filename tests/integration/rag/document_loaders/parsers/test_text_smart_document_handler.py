# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from langchain_core.documents import Document

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.text_smart_document_handler import (
    TextSmartDocumentHandler,
)

pytestmark = pytest.mark.integration


def _create_text(path: Path) -> None:

    content = """
    Intergrax Test File

    Hello from TextSmartDocumentHandler test.
    This file verifies TXT ingestion.
    """

    path.write_text(content, encoding="utf-8")


def test_text_handler_supports_extensions():

    handler = TextSmartDocumentHandler()

    assert handler.supports("file.txt") is True
    assert handler.supports("file.md") is True
    assert handler.supports("file.markdown") is True
    assert handler.supports("file.pdf") is False


def test_text_handler_confidence():

    handler = TextSmartDocumentHandler()

    assert handler.confidence("file.txt") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_text_handler_builds_parser():

    handler = TextSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) >= 1
    assert parsers[0].parser_id() == "text_loader"


def test_text_handler_loads_text(tmp_path: Path):

    text_path = tmp_path / "sample.txt"

    _create_text(text_path)

    handler = TextSmartDocumentHandler()

    docs = handler.load(str(text_path))

    assert docs
    assert all(isinstance(d, Document) for d in docs)

    content = " ".join(d.page_content for d in docs)

    assert "Hello from TextSmartDocumentHandler test." in content