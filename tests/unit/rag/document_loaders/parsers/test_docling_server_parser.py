# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from langchain_core.documents import Document

from intergrax.rag.document_loaders.parsers.docling_server_parser import DoclingServerParser
from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG, DoclingMode


pytestmark = pytest.mark.unit


class FakeResponse:
    """Fake HTTP response."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_docling_server_parser_parses_markdown(monkeypatch, tmp_path):

    def fake_post(url, **kwargs):
        return FakeResponse({"markdown": "# Title\n\nServer doc"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    file = tmp_path / "doc.pdf"
    file.write_text("dummy")

    parser = DoclingServerParser()

    docs = parser.load(str(file))

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "Server doc" in docs[0].page_content


def test_docling_server_parser_fallback_to_text(monkeypatch, tmp_path):

    def fake_post(url, **kwargs):
        return FakeResponse({"text": "Plain text content"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    file = tmp_path / "doc.pdf"
    file.write_text("dummy")

    parser = DoclingServerParser()

    docs = parser.load(str(file))

    assert docs[0].page_content == "Plain text content"


def test_docling_server_parser_handles_empty_payload(monkeypatch, tmp_path):

    def fake_post(url, **kwargs):
        return FakeResponse({})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    file = tmp_path / "doc.pdf"
    file.write_text("dummy")

    parser = DoclingServerParser()

    docs = parser.load(str(file))

    assert docs[0].page_content == ""


def test_docling_server_parser_sets_metadata(monkeypatch, tmp_path):

    def fake_post(url, **kwargs):
        return FakeResponse({"markdown": "doc"})

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    file = tmp_path / "doc.pdf"
    file.write_text("dummy")

    parser = DoclingServerParser()

    docs = parser.load(str(file))

    md = docs[0].metadata

    assert md["parser"] == "docling.server"
    assert md["position"] == 0
    assert md["source"] == str(file)


def test_docling_server_parser_is_available():
    parser = DoclingServerParser()
    assert parser.is_available() is True