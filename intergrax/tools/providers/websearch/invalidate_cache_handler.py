# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.websearch.invalidate_cache_contracts import (
    WebsearchInvalidateCacheInput,
    WebsearchInvalidateCacheOutput,
)
from intergrax.tools.providers.websearch.invalidate_cache_service import perform_websearch_invalidate_cache


class WebsearchInvalidateCacheHandler(
    ServiceToolHandler[WebsearchInvalidateCacheInput, WebsearchInvalidateCacheOutput]
):
    _service = perform_websearch_invalidate_cache
