# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.handler import ServiceToolHandler
from intergrax.tools.providers.confluence.contracts import (
    ConfluenceGetPageInput,
    ConfluencePageOutput,
    ConfluenceSearchPagesInput,
    ConfluenceSearchPagesOutput,
)
from intergrax.tools.providers.confluence.service import confluence_get_page, confluence_search_pages


class ConfluenceGetPageHandler(ServiceToolHandler[ConfluenceGetPageInput, ConfluencePageOutput]):
    _service = confluence_get_page


class ConfluenceSearchPagesHandler(
    ServiceToolHandler[ConfluenceSearchPagesInput, ConfluenceSearchPagesOutput]
):
    _service = confluence_search_pages
