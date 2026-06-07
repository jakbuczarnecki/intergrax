# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.websearch.invalidate_cache_contracts import (
    WebsearchInvalidateCacheInput,
    WebsearchInvalidateCacheOutput,
)
from intergrax.tools.registry.runtime_bindings import WebSearchCacheBinding
from intergrax.tools.registry.wiring import ToolWiringContext

WEBSEARCH_INVALIDATE_CACHE_TOOL_ID = "websearch.invalidate_cache"


def perform_websearch_invalidate_cache(
    ctx: ToolWiringContext,
    params: WebsearchInvalidateCacheInput,
) -> WebsearchInvalidateCacheOutput:
    executor = ctx.websearch_executor
    if executor is None:
        return WebsearchInvalidateCacheOutput(used=False, reason="websearch_not_configured")
    if not isinstance(executor, WebSearchCacheBinding):
        return WebsearchInvalidateCacheOutput(used=False, reason="query_cache_not_configured")

    invalidated = executor.invalidate_query_cache(
        query=params.query.strip(),
        clear_all=params.clear_all,
    )
    return WebsearchInvalidateCacheOutput(used=True, invalidated=invalidated, reason="ok")
