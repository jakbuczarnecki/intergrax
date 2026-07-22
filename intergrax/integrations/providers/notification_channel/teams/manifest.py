# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``teams`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="teams",
    categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_TEAMS',
    description='Microsoft Teams — outbound webhook notifications + inbound activities (via create_teams_integration)',
)
