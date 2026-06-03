# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``sharepoint`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="sharepoint",
    categories=(IntegrationCategory.WIKI_KNOWLEDGE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SHAREPOINT',
    description='sharepoint integration (Phase M.6 P2/P3)',
)
