# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.websearch.contracts import WebsearchQueryInput, WebsearchQueryOutput
from intergrax.tools.providers.websearch.service import perform_websearch_query


class WebsearchQueryHandler(ServiceToolHandler[WebsearchQueryInput, WebsearchQueryOutput]):
    """Tool handler for ``websearch.query``."""

    _service = perform_websearch_query
