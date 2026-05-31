# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_firecrawl_browser_automation", "register_firecrawl_integration"]

def __getattr__(name: str):
    if name == "register_firecrawl_integration":
        from intergrax.integrations.providers.browser_automation.firecrawl.register import register_firecrawl_integration
        return register_firecrawl_integration
    if name == "create_firecrawl_browser_automation":
        from intergrax.integrations.providers.browser_automation.firecrawl.bundle import create_firecrawl_browser_automation
        return create_firecrawl_browser_automation
    raise AttributeError(name)
