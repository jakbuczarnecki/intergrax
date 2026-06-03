# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``confluence`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="confluence",
    categories=(IntegrationCategory.WIKI_KNOWLEDGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_CONFLUENCE',
    description='Confluence Cloud wiki (get_page, search_pages via REST)',
)
