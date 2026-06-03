# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``selenium`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="selenium",
    categories=(IntegrationCategory.BROWSER_AUTOMATION,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SELENIUM',
    description='selenium integration (Phase M.7)',
)
