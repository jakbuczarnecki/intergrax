# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``temporal`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="temporal",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_TEMPORAL',
    description='temporal integration (Phase M.7)',
)
