# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Search logic for ``websearch.query`` — composes websearch executor or SearchProvider."""

from __future__ import annotations

from typing import Any, List, Sequence

from intergrax.tools.providers.websearch.contracts import (
    WebsearchQueryInput,
    WebsearchQueryOutput,
    WebsearchResultItem,
)
from intergrax.tools.registry.wiring import ToolWiringContext
from intergrax.websearch.schemas.web_search_result import WebSearchResult

WEBSEARCH_TOOL_ID = "websearch.query"


def perform_websearch_query(ctx: ToolWiringContext, params: WebsearchQueryInput) -> WebsearchQueryOutput:
    """
    Run web search using ``ctx.websearch_executor`` (preferred) or ``ctx.search_provider``.
    """
    if ctx.websearch_executor is not None:
        return _search_via_executor(ctx.websearch_executor, params)

    if ctx.search_provider is not None:
        return _search_via_integration(ctx.search_provider, params)

    return WebsearchQueryOutput(used=False, reason="websearch_not_configured")


def _search_via_executor(executor: Any, params: WebsearchQueryInput) -> WebsearchQueryOutput:
    raw = executor.search_sync(
        query=params.query,
        top_k=params.limit,
        locale=params.locale,
        region=params.region,
        language=params.language,
        safe_search=params.safe_search,
    )
    items = [_from_web_search_result(r) for r in raw or []]
    items = [item for item in items if item.url]
    if not items:
        return WebsearchQueryOutput(used=False, reason="no_hits")
    return WebsearchQueryOutput(
        used=True,
        results=items,
        context_text=format_websearch_context_text(items),
        reason="ok",
    )


def _search_via_integration(provider: Any, params: WebsearchQueryInput) -> WebsearchQueryOutput:
    hits = provider.search(params.query, limit=params.limit)
    items = [_from_search_hit(hit) for hit in hits or []]
    items = [item for item in items if item.url]
    if not items:
        return WebsearchQueryOutput(used=False, reason="no_hits")
    return WebsearchQueryOutput(
        used=True,
        results=items,
        context_text=format_websearch_context_text(items),
        reason="ok",
    )


def _from_web_search_result(result: WebSearchResult) -> WebsearchResultItem:
    text = (result.text or result.snippet or result.description or "").strip()
    return WebsearchResultItem(
        title=(result.title or "").strip(),
        url=(result.url or "").strip(),
        snippet=(result.snippet or "").strip(),
        text=text,
        domain=(result.domain or "").strip(),
        rank=int(result.rank or 0),
        provider=(result.provider or "").strip(),
    )


def _from_search_hit(hit: Any) -> WebsearchResultItem:
    snippet = (getattr(hit, "snippet", None) or "").strip()
    title = (getattr(hit, "title", None) or "").strip()
    url = (getattr(hit, "url", None) or "").strip()
    domain = ""
    if hasattr(hit, "domain"):
        try:
            domain = str(hit.domain() or "")
        except Exception:
            domain = ""
    return WebsearchResultItem(
        title=title,
        url=url,
        snippet=snippet,
        text=snippet or title,
        domain=domain,
        rank=int(getattr(hit, "rank", 0) or 0),
        provider=str(getattr(hit, "provider", "") or ""),
    )


def format_websearch_context_text(
    results: Sequence[WebsearchResultItem],
    *,
    max_chars: int = 4000,
) -> str:
    """Compact, LLM-friendly block of web search hits."""
    lines: List[str] = []
    total = 0

    for idx, item in enumerate(results, start=1):
        header_parts = [f"[{idx}]"]
        if item.domain:
            header_parts.append(item.domain)
        elif item.url:
            header_parts.append(item.url)

        body = item.text.strip() or item.snippet.strip() or item.title.strip()
        block = " ".join(header_parts) + "\n" + body
        if item.url:
            block += f"\nURL: {item.url}"

        if not block.strip():
            continue

        if total + len(block) + 2 > max_chars:
            remaining = max_chars - total
            if remaining > 80:
                lines.append(block[:remaining].rstrip() + "…")
            break

        lines.append(block)
        total += len(block) + 2

    return "\n\n".join(lines).strip()
