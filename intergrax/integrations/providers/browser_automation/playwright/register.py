# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register playwright."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.browser_automation.playwright.bundle import create_playwright_browser_automation
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_playwright_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.PLAYWRIGHT.value,
            categories=(IntegrationCategory.BROWSER_AUTOMATION,),
            factory=create_playwright_browser_automation,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_PLAYWRIGHT",
            description="playwright integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
