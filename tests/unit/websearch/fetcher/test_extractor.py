# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.websearch.fetcher.extractor import extract_advanced, extract_basic
from intergrax.websearch.schemas.page_content import PageContent

pytestmark = pytest.mark.unit


def _page(html: str, text: str | None = None) -> PageContent:
    return PageContent(
        final_url="https://example.com/page",
        status_code=200,
        html=html,
        text=text,
        title=None,
        description=None,
        lang=None,
    )


def test_extract_basic_sets_title_and_text() -> None:
    page = _page("<html><head><title>Hello</title></head><body><p>World</p></body></html>")
    result = extract_basic(page)
    assert result.title == "Hello"
    assert "World" in (result.text or "")


def test_extract_advanced_correct_length_before_metadata() -> None:
    page = _page(
        "<html><body><nav>skip</nav><p>Content here</p></body></html>",
        text="existing text",
    )
    result = extract_advanced(page, overwrite_existing_text=True)
    meta = (result.extra or {}).get("advanced_extraction", {})
    assert meta["length_before"] == len("existing text")
    assert meta["length_after"] == len(result.text or "")
    assert meta["overwritten_existing_text"] is True


def test_extract_advanced_boilerplate_cleanup() -> None:
    html = (
        "<html><body>"
        "<nav>nav text</nav>"
        "<aside>aside text</aside>"
        "<form><button>btn</button></form>"
        "<svg><circle/></svg>"
        "<canvas></canvas>"
        "<p>keep me</p>"
        "</body></html>"
    )
    with patch("intergrax.websearch.fetcher.extractor.HAS_TRAFILATURA", False):
        result = extract_advanced(_page(html))
    text = result.text or ""
    assert "keep me" in text
    assert "nav text" not in text
    assert "aside text" not in text
    assert "btn" not in text


def test_extract_advanced_records_extraction_method_beautifulsoup() -> None:
    html = "<html><body><p>fallback</p></body></html>"
    with patch("intergrax.websearch.fetcher.extractor.HAS_TRAFILATURA", False):
        result = extract_advanced(_page(html))
    meta = (result.extra or {}).get("advanced_extraction", {})
    assert meta["extraction_method"] == "beautifulsoup"


def test_extract_advanced_records_extraction_method_trafilatura() -> None:
    html = "<html><body><article><p>readable article text long enough</p></article></body></html>"
    mock_trafilatura = type("TrafilaturaModule", (), {})()
    mock_trafilatura.extract = lambda *args, **kwargs: "readable article text long enough"
    with patch("intergrax.websearch.fetcher.extractor.HAS_TRAFILATURA", True):
        with patch("intergrax.websearch.fetcher.extractor.trafilatura", mock_trafilatura, create=True):
            result = extract_advanced(_page(html))
    meta = (result.extra or {}).get("advanced_extraction", {})
    assert meta["extraction_method"] == "trafilatura"


def test_deterministic_whitespace_normalization() -> None:
    html = "<html><body><p>line one</p><p>line two</p></body></html>"
    with patch("intergrax.websearch.fetcher.extractor.HAS_TRAFILATURA", False):
        first = extract_advanced(_page(html))
        second = extract_advanced(_page(html))
    assert first.text == second.text
