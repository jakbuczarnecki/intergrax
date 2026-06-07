# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.wiki_knowledge import WikiKnowledge, WikiPageRecord
from intergrax.tools.providers.knowledge.contracts import (
    KnowledgeGetPageInput,
    KnowledgePageOutput,
    KnowledgeSearchInput,
    KnowledgeSearchOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

KNOWLEDGE_GET_PAGE_TOOL_ID = "knowledge.get_page"
KNOWLEDGE_SEARCH_TOOL_ID = "knowledge.search"


def _require_wiki(ctx: ToolWiringContext) -> WikiKnowledge:
    wiki = ctx.wiki_knowledge
    if wiki is None:
        raise RuntimeError("wiki_knowledge_not_configured")
    return wiki


def _to_page_output(record: WikiPageRecord) -> KnowledgePageOutput:
    return KnowledgePageOutput(
        id=record.id,
        title=record.title,
        space_key=record.space_key,
        body=record.body,
        url=record.url,
        version=record.version,
    )


def knowledge_get_page(ctx: ToolWiringContext, params: KnowledgeGetPageInput) -> KnowledgePageOutput:
    return _to_page_output(_require_wiki(ctx).get_page(params.page_id.strip()))


def knowledge_search(ctx: ToolWiringContext, params: KnowledgeSearchInput) -> KnowledgeSearchOutput:
    result = _require_wiki(ctx).search_pages(params.query.strip(), limit=params.limit)
    pages = [_to_page_output(page) for page in result.pages]
    return KnowledgeSearchOutput(pages=pages, total=int(result.total))
