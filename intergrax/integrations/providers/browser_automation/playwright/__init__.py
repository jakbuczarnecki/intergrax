# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_playwright_browser_automation", "register_playwright_integration"]

def __getattr__(name: str):
    if name == "register_playwright_integration":
        from intergrax.integrations.providers.browser_automation.playwright.register import register_playwright_integration
        return register_playwright_integration
    if name == "create_playwright_browser_automation":
        from intergrax.integrations.providers.browser_automation.playwright.bundle import create_playwright_browser_automation
        return create_playwright_browser_automation
    raise AttributeError(name)
