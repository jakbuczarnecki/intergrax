# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Optional, Dict
import httpx

from intergrax.websearch.schemas.page_content import PageContent


DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": "IntergraxWebSearch/1.0 (+https://example.com/botinfo)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


async def fetch_page(
    url: str,
    timeout: int = 20,
    headers: Optional[Dict[str, str]] = None,
    follow_redirects: bool = True,
) -> Optional[PageContent]:
    """
    Fetches a single URL and returns a PageContent instance.

    Responsibilities:
      - Perform an HTTP GET request with sane defaults.
      - Capture final URL, status code, raw HTML, and body size.
      - Keep higher-level concerns (robots, throttling, extraction) outside.

    Returns:
      PageContent on success, or None on transport-level failure.
      HTTP error codes (4xx/5xx) are represented in PageContent.status_code.
    """
    merged_headers = dict(DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    try:
        async with httpx.AsyncClient(
            headers=merged_headers,
            follow_redirects=follow_redirects,
            timeout=timeout,
        ) as client:
            response = await client.get(url)
    except (httpx.RequestError, httpx.HTTPError):
        return None

    html: Optional[str] = None
    try:
        html = response.text
    except Exception:
        html = None

    content_bytes = len(response.content) if response.content is not None else None

    page = PageContent(
        final_url=str(response.url),
        status_code=response.status_code,
        html=html,
        text=None,
        title=None,
        description=None,
        lang=None,
        og={},
        schema_org={},
        robots_allowed=None,
        content_bytes=content_bytes,
        is_paywalled=None,
        extra={
            "headers": dict(response.headers),
            "http_version": getattr(response, "http_version", None),
        },
    )
    return page
