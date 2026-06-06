# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``e2b`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="e2b",
    categories=(IntegrationCategory.SANDBOX_HOST,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_E2B',
    description='e2b integration (Phase M.6 P6)',
)
