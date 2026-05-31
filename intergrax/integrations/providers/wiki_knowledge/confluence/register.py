# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Confluence in the integration catalog (Phase M.6)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.wiki_knowledge.confluence.bundle import create_confluence_wiki_knowledge
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_confluence_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.CONFLUENCE.value,
            categories=(IntegrationCategory.WIKI_KNOWLEDGE,),
            factory=create_confluence_wiki_knowledge,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_CONFLUENCE",
            description="Confluence Cloud wiki (get_page, search_pages via REST)",
        ),
        override=override,
    )
