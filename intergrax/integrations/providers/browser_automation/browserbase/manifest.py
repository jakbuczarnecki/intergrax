# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``browserbase`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="browserbase",
    categories=(IntegrationCategory.BROWSER_AUTOMATION,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_BROWSERBASE',
    description='browserbase integration (Phase M.7 P7)',
)
