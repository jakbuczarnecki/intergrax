# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register email_smtp."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationEntry, IntegrationStatus
from intergrax.integrations.providers.notification_channel.email_smtp.bundle import create_email_smtp_notification_channel
from intergrax.integrations.registry.catalog import register_integration
from intergrax.integrations.registry.slugs import IntegrationSlug

def register_email_smtp_integration(*, override: bool = False) -> None:
    register_integration(
        IntegrationEntry(
            slug=IntegrationSlug.EMAIL_SMTP.value,
            categories=(IntegrationCategory.NOTIFICATION_CHANNEL,),
            factory=create_email_smtp_notification_channel,
            status=IntegrationStatus.BETA,
            env_prefix="INTERGRAX_EMAIL_SMTP",
            description="email_smtp integration (Phase M.6 P2/P3)",
        ),
        override=override,
    )
