# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from intergrax.tools.providers.websearch.read_url_contracts import WebsearchReadUrlInput
from intergrax.tools.providers.websearch.read_url_service import perform_websearch_read_url
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.schemas.page_content import PageContent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_websearch_read_url_extracts_text() -> None:
    page = PageContent(
        final_url="https://example.com/article",
        status_code=200,
        html="<html><head><title>Hello</title></head><body><p>World</p></body></html>",
        text="World",
        title="Hello",
        description=None,
        lang=None,
    )

    with patch(
        "intergrax.tools.providers.websearch.read_url_service.fetch_page",
        new=AsyncMock(return_value=page),
    ):
        out = perform_websearch_read_url(
            ToolWiringContext(),
            WebsearchReadUrlInput(url="https://example.com/article"),
        )

    assert out.used is True
    assert out.title == "Hello"
    assert "World" in out.text
    assert out.reason == "ok"
