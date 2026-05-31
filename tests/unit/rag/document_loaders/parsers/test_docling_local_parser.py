# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from langchain_core.documents import Document

from intergrax.integrations.contracts.document_parser import ParsedDocumentFragment
from intergrax.rag.document_loaders.parsers.docling_local_parser import DoclingLocalParser
from intergrax.rag.document_loaders.config.document_loader_config import (
    GLOBAL_DOCUMENT_LOADER_CONFIG,
    DoclingMode,
)

pytestmark = pytest.mark.unit


class _FakeDoclingBackend:
    """Stand-in for catalog ``DocumentParser`` (no docling SDK in unit tests)."""

    def __init__(self, *, text: str = "# Title\n\nTest document") -> None:
        self._text = text

    def parser_id(self) -> str:
        return "docling.local"

    def is_available(self) -> bool:
        return True

    def parse_file(self, source: str) -> list[ParsedDocumentFragment]:
        return [
            ParsedDocumentFragment(
                text=self._text,
                metadata={"source": source},
            )
        ]


@pytest.fixture
def mock_docling_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.docling_local_parser.resolve_document_parser",
        lambda slug, mode="local": _FakeDoclingBackend(),
    )


def test_docling_local_parser_returns_document(mock_docling_backend: None) -> None:
    parser = DoclingLocalParser()
    docs = parser.load("test.pdf")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Test document" in docs[0].page_content


def test_docling_local_parser_sets_metadata(mock_docling_backend: None) -> None:
    parser = DoclingLocalParser()
    docs = parser.load("test.pdf")
    metadata = docs[0].metadata

    assert metadata["parser"] == parser.parser_id()
    assert metadata["position"] == 0
    assert metadata["source"] == "test.pdf"


def test_docling_local_parser_handles_empty_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "intergrax.rag.document_loaders.parsers.docling_local_parser.resolve_document_parser",
        lambda slug, mode="local": _FakeDoclingBackend(text=""),
    )
    parser = DoclingLocalParser()
    docs = parser.load("test.pdf")

    assert len(docs) == 1
    assert docs[0].page_content == ""


def test_docling_local_parser_multiple_calls(mock_docling_backend: None) -> None:
    parser = DoclingLocalParser()
    docs1 = parser.load("a.pdf")
    docs2 = parser.load("b.pdf")

    assert docs1[0].metadata["source"] == "a.pdf"
    assert docs2[0].metadata["source"] == "b.pdf"


def test_docling_local_parser_is_available() -> None:
    mode = GLOBAL_DOCUMENT_LOADER_CONFIG.docling_mode
    parser = DoclingLocalParser()

    if mode == DoclingMode.LOCAL:
        # Depends on catalog backend availability (may be false without docling installed).
        assert isinstance(parser.is_available(), bool)
    else:
        assert parser.is_available() is False
