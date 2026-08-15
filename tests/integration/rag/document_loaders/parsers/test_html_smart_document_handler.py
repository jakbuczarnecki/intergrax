# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from pathlib import Path

from intergrax.knowledge.contracts import KnowledgeDocument, KnowledgeDocumentScope

from intergrax.rag.document_loaders.config.document_loader_config import GLOBAL_DOCUMENT_LOADER_CONFIG
from intergrax.rag.document_loaders.handlers.html_document_handler import (
    HtmlSmartDocumentHandler,
)

pytestmark = pytest.mark.integration


def _create_html(path: Path) -> None:

    html = """
    <html>
        <head>
            <title>Intergrax HTML Test</title>
        </head>
        <body>
            <h1>Intergrax</h1>
            <p>Hello from HTML handler test.</p>
        </body>
    </html>
    """

    path.write_text(html, encoding="utf-8")


def test_html_handler_supports_extensions():

    handler = HtmlSmartDocumentHandler()

    assert handler.supports("page.html") is True
    assert handler.supports("page.htm") is True
    assert handler.supports("page.txt") is False


def test_html_handler_confidence():

    handler = HtmlSmartDocumentHandler()

    assert handler.confidence("page.html") == GLOBAL_DOCUMENT_LOADER_CONFIG.default_builtin_handler_confidence


def test_html_handler_builds_parser():

    handler = HtmlSmartDocumentHandler()

    parsers = handler.build_parsers()

    assert len(parsers) >= 1


def test_html_handler_loads_html(tmp_path: Path):

    html_path = tmp_path / "sample.html"

    _create_html(html_path)

    handler = HtmlSmartDocumentHandler()

    docs = handler.load(
        str(html_path),
        scope=KnowledgeDocumentScope(tenant_id="tenant.test"),
    )

    assert docs
    assert all(isinstance(d, KnowledgeDocument) for d in docs)

    content = " ".join(d.content for d in docs)

    assert "Hello from HTML handler test." in content