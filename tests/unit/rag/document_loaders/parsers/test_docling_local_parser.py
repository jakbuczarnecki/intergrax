# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_loaders.parsers.docling_local_parser import DoclingLocalParser
from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG, DoclingMode


pytestmark = pytest.mark.unit


class FakeDoc:
    """Fake docling document object."""

    def export_to_markdown(self):
        return "# Title\n\nTest document"


class FakeResult:
    """Fake conversion result."""

    def __init__(self):
        self.document = FakeDoc()


class FakeConverter:
    """Fake DocumentConverter used for tests."""

    def convert(self, path):
        return FakeResult()


def test_docling_local_parser_returns_document(monkeypatch):

    parser = DoclingLocalParser()

    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.docling_local_parser.create_docling_converter",
        lambda: FakeConverter()
    )

    docs = parser.load("test.pdf")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Test document" in docs[0].page_content


def test_docling_local_parser_sets_metadata(monkeypatch):

    parser = DoclingLocalParser()

    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.docling_local_parser.create_docling_converter",
        lambda: FakeConverter()
    )

    docs = parser.load("test.pdf")

    metadata = docs[0].metadata

    assert metadata["parser"] == parser.parser_id()
    assert metadata["position"] == 0
    assert metadata["source"] == "test.pdf"


def test_docling_local_parser_handles_empty_text(monkeypatch):

    class EmptyDoc:
        def export_to_markdown(self):
            return ""

    class EmptyResult:
        def __init__(self):
            self.document = EmptyDoc()

    class EmptyConverter:
        def convert(self, path):
            return EmptyResult()

    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.docling_local_parser.create_docling_converter",
        lambda: EmptyConverter()
    )

    parser = DoclingLocalParser()

    docs = parser.load("test.pdf")

    assert len(docs) == 1
    assert docs[0].page_content == ""


def test_docling_local_parser_multiple_calls(monkeypatch):

    parser = DoclingLocalParser()

    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.docling_local_parser.create_docling_converter",
        lambda: FakeConverter()
    )

    docs1 = parser.load("a.pdf")
    docs2 = parser.load("b.pdf")

    assert docs1[0].metadata["source"] == "a.pdf"
    assert docs2[0].metadata["source"] == "b.pdf"


def test_docling_local_parser_is_available():

    mode = GLOBAL_DOCUMENT_LOADER_CONFIG.docling_mode

    parser = DoclingLocalParser()

    if mode == DoclingMode.LOCAL:
        assert parser.is_available() is True
    else:
        assert parser.is_available() is False