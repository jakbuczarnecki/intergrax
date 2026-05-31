# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.websearch.fetch_batch_contracts import WebsearchFetchBatchInput
from intergrax.tools.providers.websearch.fetch_batch_service import perform_websearch_fetch_batch
class WebsearchFetchBatchHandler(ServiceToolHandler[WebsearchFetchBatchInput, object]):
    _service = perform_websearch_fetch_batch
