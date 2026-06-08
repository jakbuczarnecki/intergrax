# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from intergrax.integrations.providers.browser_automation.apify.bundle import create_apify_browser_automation
from intergrax.integrations.providers.browser_automation.apify.register import register_apify_integration

__all__ = ["create_apify_browser_automation", "register_apify_integration"]
