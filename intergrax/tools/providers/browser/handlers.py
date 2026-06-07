# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.browser.contracts import BrowserFetchPageInput, BrowserFetchPageOutput
from intergrax.tools.providers.browser.service import browser_fetch_page


class BrowserFetchPageHandler(ServiceToolHandler[BrowserFetchPageInput, BrowserFetchPageOutput]):
    _service = browser_fetch_page
