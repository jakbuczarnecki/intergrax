# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

__all__ = ["create_selenium_browser_automation", "register_selenium_integration"]

def __getattr__(name: str):
    if name == "register_selenium_integration":
        from intergrax.integrations.providers.browser_automation.selenium.register import register_selenium_integration
        return register_selenium_integration
    if name == "create_selenium_browser_automation":
        from intergrax.integrations.providers.browser_automation.selenium.bundle import create_selenium_browser_automation
        return create_selenium_browser_automation
    raise AttributeError(name)
