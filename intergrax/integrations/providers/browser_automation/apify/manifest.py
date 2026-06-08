# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``apify`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="apify",
    categories=(IntegrationCategory.BROWSER_AUTOMATION,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_APIFY',
    description='apify integration (Phase M.7 P7)',
)
