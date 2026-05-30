# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.wiki_knowledge import WikiPageRecord
from intergrax.tools.providers.confluence.contracts import (
    ConfluenceGetPageInput,
    ConfluencePageOutput,
    ConfluenceSearchPagesInput,
    ConfluenceSearchPagesOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

CONFLUENCE_GET_PAGE_TOOL_ID = "confluence.get_page"
CONFLUENCE_SEARCH_PAGES_TOOL_ID = "confluence.search_pages"


def _require_wiki(ctx: ToolWiringContext):
    wiki = ctx.wiki_knowledge
    if wiki is None:
        raise RuntimeError("wiki_knowledge_not_configured")
    return wiki


def _to_page_output(record: WikiPageRecord) -> ConfluencePageOutput:
    return ConfluencePageOutput(
        id=record.id,
        title=record.title,
        space_key=record.space_key,
        body=record.body,
        url=record.url,
        version=record.version,
    )


def confluence_get_page(ctx: ToolWiringContext, params: ConfluenceGetPageInput) -> ConfluencePageOutput:
    return _to_page_output(_require_wiki(ctx).get_page(params.page_id.strip()))


def confluence_search_pages(
    ctx: ToolWiringContext,
    params: ConfluenceSearchPagesInput,
) -> ConfluenceSearchPagesOutput:
    result = _require_wiki(ctx).search_pages(params.query, limit=params.limit)
    pages = [_to_page_output(page) for page in result.pages]
    return ConfluenceSearchPagesOutput(pages=pages, total=int(result.total))
