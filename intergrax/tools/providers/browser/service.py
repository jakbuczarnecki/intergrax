# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.tools.providers.browser.contracts import BrowserFetchPageInput, BrowserFetchPageOutput
from intergrax.tools.registry.wiring import ToolWiringContext

BROWSER_FETCH_PAGE_TOOL_ID = "browser.fetch_page"


def _require_browser(ctx: ToolWiringContext) -> BrowserAutomation:
    browser = ctx.browser_automation
    if browser is None:
        raise RuntimeError("browser_automation_not_configured")
    return browser


def browser_fetch_page(ctx: ToolWiringContext, params: BrowserFetchPageInput) -> BrowserFetchPageOutput:
    page = _require_browser(ctx).fetch_page(params.url.strip(), wait_until=params.wait_until)
    return BrowserFetchPageOutput(
        url=page.url,
        title=page.title,
        text=page.text,
        status_code=page.status_code,
        html_length=len(page.html or ""),
    )
