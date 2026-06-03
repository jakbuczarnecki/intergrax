# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog manifest for ``slash_command`` integration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationStatus
from intergrax.integrations.core.manifest import IntegrationManifest

MANIFEST = IntegrationManifest(
    slug="slash_command",
    categories=(IntegrationCategory.INTERACTION_SURFACE,),
    status=IntegrationStatus.BETA,
    env_prefix='INTERGRAX_SLASH_COMMAND',
    description='Generic slash-command intake (Slack/Teams/CLI payloads)',
)
