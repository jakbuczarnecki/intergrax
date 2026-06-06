# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``hubspot`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="hubspot",
    categories=(IntegrationCategory.CRM,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_HUBSPOT',
    description='hubspot integration (Phase M.6 P6)',
)
