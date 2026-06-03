# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``pubsub`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="pubsub",
    categories=(IntegrationCategory.MESSAGE_BUS,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_PUBSUB',
    description='pubsub integration (Phase M.6 P2/P3)',
)
