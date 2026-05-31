# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import asyncio

from intergrax.tools.providers.websearch.read_url_contracts import WebsearchReadUrlInput, WebsearchReadUrlOutput
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.fetcher.extractor import extract_advanced, extract_basic
from intergrax.websearch.fetcher.http_fetcher import fetch_page

WEBSEARCH_READ_URL_TOOL_ID = "websearch.read_url"


def perform_websearch_read_url(
    _ctx: ToolWiringContext,
    params: WebsearchReadUrlInput,
) -> WebsearchReadUrlOutput:
    url = str(params.url)
    try:
        page = asyncio.run(
            fetch_page(url, timeout=int(params.timeout_seconds))
        )
    except Exception as exc:
        return WebsearchReadUrlOutput(used=False, url=url, reason=f"fetch_error:{exc.__class__.__name__}")

    if page is None:
        return WebsearchReadUrlOutput(used=False, url=url, reason="fetch_failed")

    if params.use_advanced_extraction:
        page = extract_advanced(page)
    else:
        page = extract_basic(page)

    text = (page.text or "").strip()
    if not text and page.html:
        text = (page.html or "")[:8000].strip()

    if page.status_code and page.status_code >= 400:
        return WebsearchReadUrlOutput(
            used=False,
            url=url,
            final_url=page.final_url or url,
            title=(page.title or "").strip(),
            status_code=int(page.status_code),
            reason=f"http_{page.status_code}",
        )

    if not text:
        return WebsearchReadUrlOutput(
            used=False,
            url=url,
            final_url=page.final_url or url,
            status_code=int(page.status_code or 0) or None,
            reason="empty_content",
        )

    return WebsearchReadUrlOutput(
        used=True,
        url=url,
        final_url=page.final_url or url,
        title=(page.title or "").strip(),
        text=text,
        status_code=int(page.status_code or 0) or None,
        reason="ok",
    )
