# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register log notification channel in the integration catalog (Phase M.8)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.log.bundle import create_log_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug


def register_log_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.LOG.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_log_notification_channel,
            status=IntegrationStatus.STABLE,
            env_prefix="INTERGRAX_LOG",
            description="Process-log notification channel (via create_log_integration)",
        ),
        override=override,
    )
