# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.integrations.providers.document_parser.docling.config import DoclingIntegrationConfig, DoclingMode
from intergrax.integrations.providers.document_parser.docling.opens import parse_docling_file
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.rag.document_loaders.integration.resolver import resolve_document_parser

pytestmark = pytest.mark.unit


def test_docling_server_parse_uses_parse_endpoint(monkeypatch, tmp_path) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4")

    captured: dict[str, str] = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        response = MagicMock()
        response.json.return_value = {"markdown": "# Title"}
        response.raise_for_status = MagicMock()
        return response

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)

    config = DoclingIntegrationConfig(
        mode=DoclingMode.SERVER,
        server_url="http://localhost:8000",
        server_path="/parse",
    )
    fragments = parse_docling_file(config, str(source))
    assert captured["url"] == "http://localhost:8000/parse"
    assert fragments[0].text == "# Title"


def test_resolve_docling_from_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(override=True)
    parser = resolve_document_parser("docling", mode="none")
    assert parser.parser_id() == "docling.local"
    assert parser.is_available() is False
