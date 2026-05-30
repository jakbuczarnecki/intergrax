# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.confluence.contracts import (
    ConfluenceGetPageInput,
    ConfluencePageOutput,
    ConfluenceSearchPagesInput,
    ConfluenceSearchPagesOutput,
)
from intergrax.tools.providers.confluence.service import confluence_get_page, confluence_search_pages
from intergrax.tools.registry.wiring import ToolWiringContext


class ConfluenceGetPageHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[ConfluenceGetPageInput]) -> ConfluencePageOutput:
        return confluence_get_page(self._ctx, request.input)


class ConfluenceSearchPagesHandler:
    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[ConfluenceSearchPagesInput]) -> ConfluenceSearchPagesOutput:
        return confluence_search_pages(self._ctx, request.input)
