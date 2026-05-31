# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.confluence.contracts import (
    ConfluenceGetPageInput,
    ConfluencePageOutput,
    ConfluenceSearchPagesInput,
    ConfluenceSearchPagesOutput,
)
from intergrax.tools.providers.confluence.handlers import ConfluenceGetPageHandler, ConfluenceSearchPagesHandler
from intergrax.tools.providers.confluence.service import (
    CONFLUENCE_GET_PAGE_TOOL_ID,
    CONFLUENCE_SEARCH_PAGES_TOOL_ID,
    CONFLUENCE_SEARCH_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

CONFLUENCE_BUNDLE_ID = "confluence"
CONFLUENCE_TOOL_IDS: tuple[str, ...] = (
    CONFLUENCE_GET_PAGE_TOOL_ID,
    CONFLUENCE_SEARCH_PAGES_TOOL_ID,
    CONFLUENCE_SEARCH_TOOL_ID,
)


def register_confluence_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=CONFLUENCE_GET_PAGE_TOOL_ID,
            name=CONFLUENCE_GET_PAGE_TOOL_ID,
            description="Fetch a Confluence wiki page by id (title, body, url).",
            description_short="Get Confluence page.",
            input_schema=ConfluenceGetPageInput,
            output_schema=ConfluencePageOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="wiki",
            risk_level=ToolRiskLevel.LOW,
            tags=("confluence", "wiki"),
        ),
        ConfluenceGetPageHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CONFLUENCE_SEARCH_PAGES_TOOL_ID,
            name=CONFLUENCE_SEARCH_PAGES_TOOL_ID,
            description="Search Confluence wiki pages by text query.",
            description_short="Search Confluence pages.",
            input_schema=ConfluenceSearchPagesInput,
            output_schema=ConfluenceSearchPagesOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="wiki",
            risk_level=ToolRiskLevel.LOW,
            tags=("confluence", "wiki"),
        ),
        ConfluenceSearchPagesHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=CONFLUENCE_SEARCH_TOOL_ID,
            name=CONFLUENCE_SEARCH_TOOL_ID,
            description="Search Confluence wiki pages by text query (alias of confluence.search_pages).",
            description_short="Search Confluence.",
            input_schema=ConfluenceSearchPagesInput,
            output_schema=ConfluenceSearchPagesOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="wiki",
            risk_level=ToolRiskLevel.LOW,
            tags=("confluence", "wiki"),
        ),
        ConfluenceSearchPagesHandler(ctx),
    )
