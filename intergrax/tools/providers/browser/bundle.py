# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.browser.contracts import BrowserFetchPageInput, BrowserFetchPageOutput
from intergrax.tools.providers.browser.handlers import BrowserFetchPageHandler
from intergrax.tools.providers.browser.service import BROWSER_FETCH_PAGE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

BROWSER_BUNDLE_ID = "browser"
BROWSER_TOOL_IDS: tuple[str, ...] = (BROWSER_FETCH_PAGE_TOOL_ID,)


def register_browser_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=BROWSER_FETCH_PAGE_TOOL_ID,
            name=BROWSER_FETCH_PAGE_TOOL_ID,
            description="Fetch dynamic web page content via headless browser automation.",
            description_short="Fetch page (browser).",
            input_schema=BrowserFetchPageInput,
            output_schema=BrowserFetchPageOutput,
            error_mapping={},
            side_effects=False,
            injects_context=True,
            category="browser",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("browser", "web"),
        ),
        BrowserFetchPageHandler(ctx),
    )
