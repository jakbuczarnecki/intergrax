# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register Slack in the integration catalog (Phase M.4)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.slack.bundle import create_slack_catalog_factory
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_slack_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.SLACK.value,
            categories=(
                IntegrationCategory.NOTIFICATION_CHANNEL,
                IntegrationCategory.INTERACTION_SURFACE,
            ),
            factory=create_slack_catalog_factory,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_SLACK",
            description=(
                "Slack — outbound webhook notifications + inbound slash commands "
                "(via create_slack_integration)"
            ),
        ),
        override=override,
    )
