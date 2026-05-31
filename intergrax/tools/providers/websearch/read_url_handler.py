# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.websearch.read_url_contracts import WebsearchReadUrlInput, WebsearchReadUrlOutput
from intergrax.tools.providers.websearch.read_url_service import perform_websearch_read_url


class WebsearchReadUrlHandler(ServiceToolHandler[WebsearchReadUrlInput, WebsearchReadUrlOutput]):
    _service = perform_websearch_read_url
