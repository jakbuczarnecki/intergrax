# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``slack`` conversation channel integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="slack",
    categories=(IntegrationCategory.CONVERSATION_CHANNEL,),
    status=IntegrationStatus.BETA,
    env_prefix="INTERGRAX_SLACK",
    description=(
        "Slack conversation channel with Socket Mode inbound and Web API outbound runtime"
    ),
)
