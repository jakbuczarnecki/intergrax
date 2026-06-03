# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``serpapi`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="serpapi",
    categories=(IntegrationCategory.SEARCH_PROVIDER,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SERPAPI',
    description='serpapi integration (Phase M.6 P2/P3)',
)
