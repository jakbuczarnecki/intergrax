# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``firecrawl`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="firecrawl",
    categories=(IntegrationCategory.BROWSER_AUTOMATION,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_FIRECRAWL',
    description='firecrawl integration (Phase M.7)',
)
