# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``service_bus`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="service_bus",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SERVICE_BUS',
    description='service_bus integration (Phase M.6 P2/P3)',
)
