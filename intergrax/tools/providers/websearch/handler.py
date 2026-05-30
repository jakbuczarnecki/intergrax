# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.providers.websearch.contracts import WebsearchQueryInput, WebsearchQueryOutput
from intergrax.tools.providers.websearch.service import perform_websearch_query
from intergrax.tools.registry.wiring import ToolWiringContext


class WebsearchQueryHandler:
    """Tool handler for ``websearch.query``."""

    def __init__(self, ctx: ToolWiringContext) -> None:
        self._ctx = ctx

    def execute(self, request: ToolExecutionRequest[WebsearchQueryInput]) -> WebsearchQueryOutput:
        return perform_websearch_query(self._ctx, request.input)
