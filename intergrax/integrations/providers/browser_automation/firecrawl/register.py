# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register firecrawl."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.browser_automation.firecrawl.bundle import create_firecrawl_browser_automation
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_firecrawl_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.FIRECRAWL.value,
            categories=(IntegrationCategory.BROWSER_AUTOMATION,),
            factory=create_firecrawl_browser_automation,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_FIRECRAWL",
            description="firecrawl integration (Phase M.7)",
        ),
        override=override,
    )
