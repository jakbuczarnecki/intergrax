# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.websearch.fetch_batch_contracts import (
    WebsearchFetchBatchInput,
    WebsearchFetchBatchOutput,
    WebsearchFetchBatchPageOutput,
)
from intergrax.tools.providers.websearch.read_url_contracts import WebsearchReadUrlInput
from intergrax.tools.providers.websearch.read_url_service import perform_websearch_read_url
from intergrax.tools.registry.wiring import ToolWiringContext

WEBSEARCH_FETCH_BATCH_TOOL_ID = "websearch.fetch_batch"


def perform_websearch_fetch_batch(
    ctx: ToolWiringContext,
    params: WebsearchFetchBatchInput,
) -> WebsearchFetchBatchOutput:
    pages: list[WebsearchFetchBatchPageOutput] = []
    success_count = 0
    context_parts: list[str] = []

    for url in params.urls:
        single = perform_websearch_read_url(
            ctx,
            WebsearchReadUrlInput(
                url=str(url),
                timeout_seconds=params.timeout_seconds,
                use_advanced_extraction=params.use_advanced_extraction,
            ),
        )
        page = WebsearchFetchBatchPageOutput(
            url=single.url,
            final_url=single.final_url,
            title=single.title or "",
            text=single.text or "",
            status_code=single.status_code,
            used=single.used,
            reason=single.reason or "",
        )
        pages.append(page)
        if single.used:
            success_count += 1
            header = single.title or single.final_url or single.url
            snippet = (single.text or "")[:1500]
            context_parts.append(f"### {header}\n{snippet}")

    return WebsearchFetchBatchOutput(
        pages=pages,
        success_count=success_count,
        context_text="\n\n".join(context_parts).strip(),
    )
