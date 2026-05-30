# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register websearch catalog tools."""

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.websearch.contracts import WebsearchQueryInput, WebsearchQueryOutput
from intergrax.tools.providers.websearch.handler import WebsearchQueryHandler
from intergrax.tools.providers.websearch.service import WEBSEARCH_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

WEBSEARCH_BUNDLE_ID = "websearch"
WEBSEARCH_TOOL_IDS: tuple[str, ...] = (WEBSEARCH_TOOL_ID,)


def websearch_query_contract() -> ToolContract:
    return ToolContract(
        tool_id=WEBSEARCH_TOOL_ID,
        name="websearch.query",
        description=(
            "Search the public web for up-to-date information. Use for current events, "
            "vendor documentation, market data, or facts not present in indexed documents. "
            "Returns ranked titles, URLs, snippets, and a compact context block."
        ),
        description_short="Search the web for current information.",
        input_schema=WebsearchQueryInput,
        output_schema=WebsearchQueryOutput,
        error_mapping={},
        side_effects=False,
        injects_context=True,
        category="retrieval",
        risk_level=ToolRiskLevel.LOW,
        tags=("websearch", "retrieval", "context"),
    )


def register_websearch_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    handler = WebsearchQueryHandler(ctx)
    registry.register(websearch_query_contract(), handler)
