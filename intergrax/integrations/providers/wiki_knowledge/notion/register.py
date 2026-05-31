# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register notion."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.wiki_knowledge.notion.bundle import create_notion_wiki_knowledge
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_notion_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.NOTION.value,
            categories=(IntegrationCategory.WIKI_KNOWLEDGE,),
            factory=create_notion_wiki_knowledge,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_NOTION",
            description="notion integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
