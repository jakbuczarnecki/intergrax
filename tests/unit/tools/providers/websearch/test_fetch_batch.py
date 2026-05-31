# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from intergrax.tools.providers.websearch.fetch_batch_contracts import WebsearchFetchBatchInput
from intergrax.tools.providers.websearch.fetch_batch_service import perform_websearch_fetch_batch
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.schemas.page_content import PageContent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_websearch_fetch_batch_returns_multiple_pages() -> None:
    page = PageContent(
        final_url="https://example.com/a",
        status_code=200,
        html="<p>A</p>",
        text="Alpha",
        title="A",
        description=None,
        lang=None,
    )

    with patch(
        "intergrax.tools.providers.websearch.read_url_service.fetch_page",
        new=AsyncMock(return_value=page),
    ):
        out = perform_websearch_fetch_batch(
            ToolWiringContext(),
            WebsearchFetchBatchInput(urls=["https://example.com/a", "https://example.com/b"]),
        )

    assert out.success_count == 2
    assert len(out.pages) == 2
    assert "Alpha" in out.context_text
