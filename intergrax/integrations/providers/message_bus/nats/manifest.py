# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``nats`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="nats",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_NATS',
    description='nats integration (Phase M.7)',
)
