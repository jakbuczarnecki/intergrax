# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register websearch catalog tools."""

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.websearch.contracts import WebsearchQueryInput, WebsearchQueryOutput
from intergrax.tools.providers.websearch.handler import WebsearchQueryHandler
from intergrax.tools.providers.websearch.read_url_contracts import WebsearchReadUrlInput, WebsearchReadUrlOutput
from intergrax.tools.providers.websearch.read_url_handler import WebsearchReadUrlHandler
from intergrax.tools.providers.websearch.read_url_service import WEBSEARCH_READ_URL_TOOL_ID
from intergrax.tools.providers.websearch.service import WEBSEARCH_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

WEBSEARCH_BUNDLE_ID = "websearch"
WEBSEARCH_TOOL_IDS: tuple[str, ...] = (WEBSEARCH_TOOL_ID, WEBSEARCH_READ_URL_TOOL_ID)


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


def websearch_read_url_contract() -> ToolContract:
    return ToolContract(
        tool_id=WEBSEARCH_READ_URL_TOOL_ID,
        name="websearch.read_url",
        description=(
            "Fetch and extract readable text from a specific HTTP(S) URL. "
            "Use after websearch.query when full page content is needed."
        ),
        description_short="Fetch and extract text from a URL.",
        input_schema=WebsearchReadUrlInput,
        output_schema=WebsearchReadUrlOutput,
        error_mapping={},
        side_effects=False,
        injects_context=True,
        category="retrieval",
        risk_level=ToolRiskLevel.LOW,
        tags=("websearch", "fetch", "context"),
    )


def register_websearch_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(websearch_query_contract(), WebsearchQueryHandler(ctx))
    registry.register(websearch_read_url_contract(), WebsearchReadUrlHandler(ctx))
