# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register selenium."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.browser_automation.selenium.bundle import create_selenium_browser_automation
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_selenium_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SELENIUM.value,
            categories=(IntegrationCategory.BROWSER_AUTOMATION,),
            factory=create_selenium_browser_automation,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SELENIUM",
            description="selenium integration (Phase M.7)",
        ),
        override=override,
    )
