# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.integrations.contracts.browser_automation import BrowserAutomation
from intergrax.integrations.contracts.browser_automation_navigation import (
    BrowserAutomationNavigationTimeoutPort,
)
from intergrax.tools.providers.browser.contracts import BrowserFetchPageInput, BrowserFetchPageOutput
from intergrax.tools.providers.provider_resource_enforcement import truncate_utf8_chars
from intergrax.tools.registry.wiring import ToolWiringContext

BROWSER_FETCH_PAGE_TOOL_ID = "browser.fetch_page"


def _require_browser(ctx: ToolWiringContext) -> BrowserAutomation:
    browser = ctx.browser_automation
    if browser is None:
        raise RuntimeError("browser_automation_not_configured")
    return browser


def browser_fetch_page(ctx: ToolWiringContext, params: BrowserFetchPageInput) -> BrowserFetchPageOutput:
    browser = _require_browser(ctx)
    timeout_enforced = isinstance(browser, BrowserAutomationNavigationTimeoutPort)
    page = browser.fetch_page(
        params.url.strip(),
        wait_until=params.wait_until,
        navigation_timeout_s=params.navigation_timeout_s,
    )
    text, truncated = truncate_utf8_chars(page.text, params.max_text_chars)
    return BrowserFetchPageOutput(
        url=page.url,
        title=page.title,
        text=text,
        status_code=page.status_code,
        html_length=len(page.html or ""),
        truncated=truncated,
        navigation_timeout_enforced=timeout_enforced,
    )
