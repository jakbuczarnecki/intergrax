# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``influxdb`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="influxdb",
    categories=(IntegrationCategory.OBSERVABILITY_BACKEND,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_INFLUXDB',
    description='influxdb integration (Phase M.6 P4)',
)
