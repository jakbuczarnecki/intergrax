# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``playwright`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="playwright",
    categories=(IntegrationCategory.BROWSER_AUTOMATION,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_PLAYWRIGHT',
    description='playwright integration (Phase M.6 P2/P3)',
)
