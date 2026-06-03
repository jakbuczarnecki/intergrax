# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``slack`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="slack",
    categories=(IntegrationCategory.NOTIFICATION_CHANNEL, IntegrationCategory.INTERACTION_SURFACE,),
    status=IntegrationStatus.STABLE,
    env_prefix='INTERGRAX_SLACK',
    description='Slack — outbound webhook notifications + inbound slash commands (via create_slack_integration)',
)
