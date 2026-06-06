# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``pulsar`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="pulsar",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_PULSAR',
    description='pulsar integration (Phase M.6 P6)',
)
