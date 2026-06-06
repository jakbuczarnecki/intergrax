# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``incident_io`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="incident_io",
    categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_INCIDENT_IO',
    description='incident_io integration (Phase M.6 P4)',
)
