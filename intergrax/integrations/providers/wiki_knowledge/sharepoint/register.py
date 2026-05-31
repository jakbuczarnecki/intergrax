# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register sharepoint."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.wiki_knowledge.sharepoint.bundle import create_sharepoint_wiki_knowledge
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_sharepoint_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SHAREPOINT.value,
            categories=(IntegrationCategory.WIKI_KNOWLEDGE,),
            factory=create_sharepoint_wiki_knowledge,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_SHAREPOINT",
            description="sharepoint integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
